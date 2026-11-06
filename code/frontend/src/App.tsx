import type { FloorPlan, PlanPoint, Room } from './types'
import { useEffect, useMemo, useState } from 'react'
import { WorldProvider, useWorld } from './state/WorldContext'
import { WorldView } from './components/WorldView'
import { TopBar } from './components/TopBar'
import { EventDrawer } from './components/EventDrawer'
import { HeartbeatPreview } from './components/HeartbeatPreview'
import { OracleInput } from './components/OracleInput'
import { StatusNotice } from './components/StatusNotice'
import { BuildTray } from './components/BuildTray'
import { PersonaRevisionModal } from './components/PersonaRevisionModal'
import { AgentStatusModal } from './components/AgentStatusModal'
import { RoomPreview } from './components/RoomPreview'
import { useIdleHeartbeat } from './hooks/useIdleHeartbeat'
import { PALETTE_AGENTS, AMBIENT_LINES, type PaletteRoom } from './mock/data'
import { DRAWER_AUTO_OPEN_MS } from './config/tuning'
import { DEFAULT_FLOOR_PLAN } from './layouts/defaultFloorPlan'
import { cloneFloorPlan, makeRoomModule, syncPlanShell } from './layouts/floorPlanEditing'

import './styles/global.css'
import './styles/pixel.css'
import './styles/animations.css'
import './styles/world.css'
import './styles/floorplan.css'
import './styles/ui.css'

function usableFloorPlan(plan: FloorPlan | null | undefined): FloorPlan {
  // RoomKind 已由类型系统约束，无需运行时白名单校验。
  // 仅做最小合法性检查：plan 存在且有房间即采用，否则回退默认户型。
  // 早期此处硬编码了 living/kitchen/bedroom/study 白名单，会把自定义户型里
  // 合法的 entry/bath/balcony 误判为无效 → 保存后回退默认，户型丢失。
  if (!plan || plan.rooms.length === 0) return DEFAULT_FLOOR_PLAN
  return plan
}

function Home() {
  const {
    state,
    bubbles,
    notice,
    setMode,
    setDrawer,
    toggleDrawer,
    castOracle,
    moveAgent,
    addAgentTo,
    deleteAgent,
    updateFloorPlan,
    advancePhase,
    regenerateOpening,
    confirmPersonaRevision,
    emitAmbient,
  } = useWorld()

  const { world, mode, drawerOpen, lastInteractionTs, oraclePending, catchingUp, agentPositions } = state
  const buildMode = mode === 'build'
  const [personaRevisionOpen, setPersonaRevisionOpen] = useState(false)
  const [inspectedAgentId, setInspectedAgentId] = useState<string | null>(null)
  const [previewRoomId, setPreviewRoomId] = useState<string | null>(null)
  const inspectedAgent = inspectedAgentId
    ? world.agents.find((agent) => agent.id === inspectedAgentId)
    : null
  const baseFloorPlan = usableFloorPlan(world.floorPlan)
  const [draftFloorPlan, setDraftFloorPlan] = useState<FloorPlan>(() => cloneFloorPlan(baseFloorPlan))
  const [selectedPlanRoomId, setSelectedPlanRoomId] = useState<string | null>(null)
  const [selectedBuildAgentId, setSelectedBuildAgentId] = useState<string | null>(null)
  const [addedRooms, setAddedRooms] = useState<Room[]>([])
  const [savingFloorPlan, setSavingFloorPlan] = useState(false)
  const displayedFloorPlan = buildMode ? draftFloorPlan : baseFloorPlan
  const previewRoom = previewRoomId
    ? world.rooms.find((room) => room.id === previewRoomId)
    : null
  const previewPlanRoom = previewRoomId
    ? displayedFloorPlan.rooms.find((room) => room.roomId === previewRoomId)
    : null
  const previewAgents = previewRoomId
    ? world.agents.filter((agent) => agent.roomId === previewRoomId)
    : []
  const selectedPlanRoom = draftFloorPlan.rooms.find((room) => room.id === selectedPlanRoomId) ?? null
  const selectedBuildAgent = selectedBuildAgentId
    ? world.agents.find((agent) => agent.id === selectedBuildAgentId)
    : null
  const selectedModuleName = selectedBuildAgent?.name ?? selectedPlanRoom?.name ?? null
  const selectedRoomCanDelete = Boolean(
    selectedBuildAgent || (selectedPlanRoom && draftFloorPlan.rooms.length > 1),
  )
  const floorPlanDirty = useMemo(
    () => addedRooms.length > 0 || JSON.stringify(draftFloorPlan) !== JSON.stringify(baseFloorPlan),
    [addedRooms.length, draftFloorPlan, baseFloorPlan],
  )

  useEffect(() => {
    if (buildMode) return
    setDraftFloorPlan(cloneFloorPlan(baseFloorPlan))
    setSelectedPlanRoomId(null)
    setAddedRooms([])
  }, [buildMode, baseFloorPlan])

  // 空闲心跳引擎：拼搭模式下暂停（避免干扰操作），生活模式常开
  // inventory 取后端随 world 下发的氛围库存；离线/未下发时兜底本地 AMBIENT_LINES。
  useIdleHeartbeat({
    lastInteractionTs,
    enabled: mode === 'live',
    onAmbient: emitAmbient,
    inventory: world.ambientInventory ?? AMBIENT_LINES,
  })

  // 首次进门：自动滑出一次事件流，几秒后自动收起（PRD 9.10 / design.md 6.3）
  // 依赖 [setDrawer] 稳定，仅在挂载时运行；StrictMode 双调用由 cleanup 兜住，
  // 不用 ref 守卫（那会导致第二次挂载跳过关闭定时器 → 抽屉不收起）。
  useEffect(() => {
    setDrawer(true)
    const t = window.setTimeout(() => setDrawer(false), DRAWER_AUTO_OPEN_MS)
    return () => window.clearTimeout(t)
  }, [setDrawer])

  // 托盘 drop：把 paletteId 解析成具体项后交给 context。
  // 户型模板已经提供完整空间，本阶段只支持拖成员进具体空间，不再把“房间”拖到空格。
  const handleDropPaletteAgent = (paletteId: string, roomId: string, pos: PlanPoint) => {
    const item = PALETTE_AGENTS.find((a) => a.paletteId === paletteId)
    if (item) addAgentTo(item, roomId, pos)
  }

  const handleEnterBuild = () => {
    setDraftFloorPlan(cloneFloorPlan(baseFloorPlan))
    setAddedRooms([])
    setSelectedPlanRoomId(null)
    setSelectedBuildAgentId(null)
    setMode('build')
  }

  const handleAddRoomModule = (item: PaletteRoom) => {
    const sameKindCount = [...world.rooms, ...addedRooms].filter((room) => room.kind === item.kind).length
    const { planRoom, room } = makeRoomModule(item, draftFloorPlan, sameKindCount)
    setDraftFloorPlan((prev) => syncPlanShell({ ...prev, rooms: [...prev.rooms, planRoom] }))
    setAddedRooms((prev) => [...prev, room])
    setSelectedPlanRoomId(planRoom.id)
    setSelectedBuildAgentId(null)
  }

  const handleResetFloorPlan = () => {
    setDraftFloorPlan(cloneFloorPlan(baseFloorPlan))
    setAddedRooms([])
    setSelectedPlanRoomId(null)
    setSelectedBuildAgentId(null)
  }

  const handleDeleteSelectedModule = () => {
    if (selectedBuildAgent) {
      void deleteAgent(selectedBuildAgent.id)
      setSelectedBuildAgentId(null)
      return
    }
    if (!selectedPlanRoom || !selectedRoomCanDelete) return
    setDraftFloorPlan((prev) =>
      syncPlanShell({
        ...prev,
        rooms: prev.rooms.filter((room) => room.id !== selectedPlanRoom.id),
      }),
    )
    setAddedRooms((prev) => prev.filter((room) => room.id !== selectedPlanRoom.roomId))
    setSelectedPlanRoomId(null)
  }

  const handleSaveFloorPlan = async (exitAfterSave = false) => {
    if (savingFloorPlan) return
    if (!floorPlanDirty) {
      if (exitAfterSave) {
        setMode('live')
        regenerateOpening()
      }
      return
    }
    setSavingFloorPlan(true)
    try {
      const accepted = await updateFloorPlan(draftFloorPlan, addedRooms)
      if (accepted) {
        setAddedRooms([])
        if (exitAfterSave) {
          setMode('live')
          regenerateOpening()
        }
      }
    } finally {
      setSavingFloorPlan(false)
    }
  }

  const latestEvent = world.events[world.events.length - 1]

  return (
    <>
      <WorldView
        world={world}
        floorPlan={displayedFloorPlan}
        bubbles={bubbles}
        buildMode={buildMode}
        agentPositions={agentPositions}
        onMoveAgent={moveAgent}
        onDropPaletteAgent={handleDropPaletteAgent}
        onInspectAgent={setInspectedAgentId}
        onInspectRoom={setPreviewRoomId}
        onSelectBuildAgent={(agentId) => {
          setSelectedBuildAgentId(agentId)
          setSelectedPlanRoomId(null)
        }}
        selectedPlanRoomId={selectedPlanRoomId}
        onSelectPlanRoom={(planRoomId) => {
          setSelectedPlanRoomId(planRoomId)
          setSelectedBuildAgentId(null)
        }}
        onChangePlan={setDraftFloorPlan}
      />

      <TopBar
        world={world}
        onToggleDrawer={toggleDrawer}
        onEnterBuild={handleEnterBuild}
        onAdvancePhase={advancePhase}
      />

      {/* 生活模式专属：神谕输入 + 心跳预告 */}
      {!buildMode && (
        <>
          <OracleInput onCast={castOracle} pending={oraclePending} />
          <HeartbeatPreview latest={latestEvent} visible={!drawerOpen} />
        </>
      )}

      {/* 拼搭模式专属：改造条 + 底部托盘 */}
      {/* 「✓ 完成」离开拼搭回生活模式时，按当前（可能已换角色/布局的）世界重新生成开场，
          让改造后的家用一段新叙事迎接你。开场失败静默略过（见 regenerateOpening）。 */}
      {buildMode && (
        <BuildTray
          floorPlanDirty={floorPlanDirty}
          savingFloorPlan={savingFloorPlan}
          selectedRoomName={
            selectedModuleName
          }
          selectedRoomCanDelete={selectedRoomCanDelete}
          onAddRoomModule={handleAddRoomModule}
          onDeleteSelectedRoomModule={handleDeleteSelectedModule}
          onSaveFloorPlan={() => void handleSaveFloorPlan(false)}
          onResetFloorPlan={handleResetFloorPlan}
          onDone={() => void handleSaveFloorPlan(true)}
        />
      )}

      {/* 沉浸式轻量提示（离线/异常/无反应）——两种模式都可见，不破坏沉浸 */}
      <StatusNotice notice={notice} />

      {/* F10 进门补演加载态：低调显示「正在推门进家…」，补完即隐。不用进度条/数值（红线）。 */}
      {catchingUp && (
        <div className="catchup-notice" role="status" aria-live="polite">
          <span className="catchup-dot" aria-hidden />
          <span className="catchup-text">正在推门进家…</span>
        </div>
      )}

      {/* 事件流抽屉：两种模式都可查看。传 agents 供反应/走动事件补回「名字：」归属 */}
      <EventDrawer
        open={drawerOpen}
        events={world.events}
        agents={world.agents}
        onManageAgents={() => setPersonaRevisionOpen(true)}
        onClose={() => setDrawer(false)}
      />

      {personaRevisionOpen && (
        <PersonaRevisionModal
          agents={world.agents}
          rooms={world.rooms}
          onCancel={() => setPersonaRevisionOpen(false)}
          onConfirm={confirmPersonaRevision}
        />
      )}

      {inspectedAgent && !buildMode && (
        <AgentStatusModal
          agent={inspectedAgent}
          rooms={world.rooms}
          onCancel={() => setInspectedAgentId(null)}
          onConfirm={confirmPersonaRevision}
        />
      )}

      {/* 星露谷式房间预览：点击房间 → 全屏进入房间内部（独立大图 + 房间内角色） */}
      {previewRoom && previewPlanRoom && !buildMode && (
        <RoomPreview
          room={previewRoom}
          kind={previewPlanRoom.kind}
          agents={previewAgents}
          bubbles={bubbles}
          onInspectAgent={setInspectedAgentId}
          onClose={() => setPreviewRoomId(null)}
        />
      )}
    </>
  )
}

export default function App() {
  // 首帧即用本地种子世界渲染（保证「一进门就是活的」）；WorldProvider 挂载后
  // 异步 GET /world 覆盖或确认，失败则显式离线兜底（见 WorldContext）。无需 boot 空等。
  return (
    <WorldProvider>
      <Home />
    </WorldProvider>
  )
}
