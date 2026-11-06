// CyberHome 核心领域类型
// 说明：情绪/关系「藏在言行里」（PRD 9.11），因此 Mood 只用于内部驱动措辞，
// 绝不渲染成数值条/进度条。

/** 世界时间阶段（连续漂移的光照由它驱动） */
export type WorldPhase = 'day' | 'dusk' | 'night' | 'lateNight'

/** 房间类型（预制模块，MVP 有限几种） */
export type RoomKind = 'living' | 'kitchen' | 'bedroom' | 'study' | 'entry' | 'bath' | 'balcony'

/** 角色类型：人 / 家电（万物皆 Agent，平权） */
export type AgentKind = 'human' | 'appliance'

/** 心情：仅用于内部生成措辞，不做面板可视化 */
export type Mood = 'happy' | 'calm' | 'tired' | 'grumpy' | 'lonely'

/** 双模式：生活（沉浸）/ 拼搭（改造） */
export type WorldMode = 'live' | 'build'

/** 网格坐标（1-based，直接映射 CSS grid-column / grid-row） */
export interface GridPos {
  col: number
  row: number
}

export interface Room {
  id: string
  kind: RoomKind
  name: string
  /** 房间标志 emoji（作为墙上的牌子） */
  emoji: string
  pos: GridPos
  /** 房间主题色（边框/底色点缀） */
  accent: string
}

export interface Agent {
  id: string
  kind: AgentKind
  name: string
  /** 外观种子：兼容旧 emoji 数据，渲染层转成像素形象 */
  avatar: string
  /** 一句话人设 */
  persona: string
  /** 当前内心独白（常驻低调小气泡，保证「一进门就是活的」） */
  monologue: string
  mood: Mood
  /** 口头禅（纯数据，用于 mock 生成个性化反应；新拖入的角色也自带） */
  catchphrase?: string
  /** 当前所在房间 id */
  roomId: string
  /** 可选真实素材图；缺省时继续使用像素形象 */
  asset?: string
}

/** 事件来源，用于事件流样式区分与调试 */
export type EventSource =
  | 'opening' // 预置开场故事
  | 'oracle' // 神谕投放本身
  | 'reaction' // 角色对神谕的反应
  | 'move' // 拖角色进房间触发
  | 'heartbeat' // 空闲氛围心跳
  | 'time' // 世界时间推进
  | 'catchup' // 惰性补演追溯（后端专有；离开再回来时补写这段的生活）
  | 'system' // 系统级提示（如离线/异常的世界口吻旁白）

export interface WorldEvent {
  id: string
  ts: number
  text: string
  source: EventSource
  /** 世界时刻快照（如「傍晚」），显示用，不显示精确时间 */
  clockLabel?: string
  /** 关联角色（可选） */
  agentId?: string
  /** Async generated events should display after this source event. */
  anchorEventId?: string | null
}

export interface World {
  id: string
  name: string
  phase: WorldPhase
  /** 当前世界时钟展示串，如「周三 傍晚 18:40」 */
  clockLabel: string
  gridCols: number
  gridRows: number
  rooms: Room[]
  agents: Agent[]
  /** 事件流，按时间正序存储（旧→新），展示时倒序 */
  events: WorldEvent[]
  /** 自定义户型图。缺省/null 时使用 DEFAULT_FLOOR_PLAN。 */
  floorPlan?: FloorPlan | null
  /**
   * 空闲心跳氛围文案库存（F11）：后端随 world 下发，前端只消费不实时烧 token。
   * 可选——离线 mock 世界也带（见 mock/data.ts），缺省时前端 fallback 到 AMBIENT_LINES。
   */
  ambientInventory?: string[]
}

/** 户型图坐标：使用模板内部坐标系，渲染时由 SVG viewBox 自适应缩放。 */
export interface PlanPoint {
  x: number
  y: number
}

export type FloorMaterial = 'oak' | 'tatami' | 'tile' | 'stone' | 'sunroom'

export interface FloorPlanRoom {
  /** 户型模板内部 id，稳定用于 React key 与样式钩子 */
  id: string
  /** 关联世界房间 id。世界状态仍只写 roomId，不写自由坐标。 */
  roomId: string
  kind: RoomKind
  name: string
  floor: FloorMaterial
  polygon: PlanPoint[]
  labelAnchor: PlanPoint
  agentAnchors: PlanPoint[]
}

export interface WallSegment {
  id: string
  from: PlanPoint
  to: PlanPoint
  thickness: number
}

export interface DoorSegment {
  id: string
  roomIds: [string, string]
  from: PlanPoint
  to: PlanPoint
  swing: 'left' | 'right' | 'none'
}

export interface WindowSegment {
  id: string
  roomId: string
  from: PlanPoint
  to: PlanPoint
}

export interface FloorPlan {
  id: string
  name: string
  width: number
  height: number
  rooms: FloorPlanRoom[]
  walls: WallSegment[]
  doors: DoorSegment[]
  windows: WindowSegment[]
}

/**
 * what-if 只读预演的触发描述（F12）。对齐后端 WhatIfRequest：
 * 「假如在某一刻（seq）换一条触发（kind + 载荷）会怎样」。
 * seq 省略=从最新时刻岔开，-1=从最初岔开；纯前端概念上不向用户裸露数字。
 */
export interface WhatIfTrigger {
  kind: 'oracle' | 'move'
  text?: string
  agentId?: string
  roomId?: string
  seq?: number
}
