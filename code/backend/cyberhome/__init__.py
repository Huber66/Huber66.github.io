"""CyberHome 后端 —— 生成式多 Agent 引擎（决策 24）。

模块地图：
  models     领域模型（frozen dataclass，对齐 frontend/src/types.ts）
  seed       种子世界（对齐 src/mock/data.ts）
  engine     纯函数内核 step()
  reactor    共处涌现：Reactor 抽象 + StubReactor（可复现）+ LLMReactor（真 LLM，F06）
  memory     MemoryStore 接口 + 内存 stub + SQLite/Mem0 后端 + 回退工厂（F13/F09）
  config     LLM env 配置（DeepSeek / MiMo / stub 自动选择，F04）
  llm        ChatModel 抽象 + provider adapters + stub（无 key 兜底，F04）
  prompting  私有视角 prompt 组装 + 上下文预算 + 反应解析（F05）
  personas   引导问答 → LLM 合成角色草稿（无 key 退确定性 stub，F07b）
  opening    读当前世界动态生成开场故事（无 key 退确定性 stub，F08）
  eventstore 事件溯源：回放/回溯/what-if（决策 32/33）
  persistence SQLite 持久化：世界快照 + 事件溯源日志 + last_seen（F03）
  driver     把 step() 与 EventStore 串起来的应用层
"""

from .models import Agent, Mood, Room, Trigger, World, WorldEvent, WorldPatch, apply_patch
from .engine import step, agents_in_room
from .eventstore import EventStore, LogEntry
from .memory import (
    InMemoryStore,
    Mem0MemoryStore,
    MemoryItem,
    MemoryStore,
    SQLiteMemoryStore,
    build_memory_store,
    resolve_chroma_path,
    resolve_memory_db_path,
)
from .config import LLMConfig, load_llm_config
from .llm import (
    ChatModel,
    DeepSeekChatModel,
    LLMResponse,
    MiMoChatModel,
    StubChatModel,
    build_chat_model,
    parse_json_response,
)
from .prompting import (
    AssembledPrompt,
    ReactionDecision,
    assemble_prompt,
    parse_reaction,
)
from .personas import (
    PersonaAnswers,
    PersonaDraft,
    coerce_mood,
    stub_persona_draft,
    synthesize_persona_draft,
)
from .persona_revision import (
    PersonaRevisionAnswers,
    PersonaRevisionDraft,
    stub_persona_revision_draft,
    synthesize_persona_revision_draft,
)
from .opening import (
    OpeningBeat,
    stub_opening,
    synthesize_opening,
)
from .persistence import (
    PersistedEntry,
    PersistenceStore,
    SQLitePersistenceStore,
    resolve_db_path,
)
from .reactor import AgentReaction, LLMReactor, Reactor, StubReactor
from .seed import make_initial_world, new_event_id
from .worldclock import advance_phase_label, next_phase, phase_label, phase_short
from .driver import WorldDriver

__all__ = [
    # models
    "Agent", "Mood", "Room", "Trigger", "World", "WorldEvent", "WorldPatch", "apply_patch",
    # engine
    "step", "agents_in_room",
    # eventstore
    "EventStore", "LogEntry",
    # memory
    "InMemoryStore", "MemoryStore", "MemoryItem",
    "SQLiteMemoryStore", "Mem0MemoryStore",
    "build_memory_store", "resolve_chroma_path", "resolve_memory_db_path",
    # config / llm
    "LLMConfig", "load_llm_config",
    "ChatModel", "LLMResponse", "StubChatModel", "DeepSeekChatModel", "MiMoChatModel",
    "build_chat_model", "parse_json_response",
    # prompting
    "AssembledPrompt", "ReactionDecision", "assemble_prompt", "parse_reaction",
    # personas (F07b 创建漏斗合成)
    "PersonaAnswers", "PersonaDraft", "coerce_mood", "stub_persona_draft", "synthesize_persona_draft",
    # persona revision (P0-1 人设修订)
    "PersonaRevisionAnswers", "PersonaRevisionDraft",
    "stub_persona_revision_draft", "synthesize_persona_revision_draft",
    # opening (F08 开场故事生成)
    "OpeningBeat", "stub_opening", "synthesize_opening",
    # persistence
    "PersistedEntry", "PersistenceStore", "SQLitePersistenceStore", "resolve_db_path",
    # reactor
    "Reactor", "StubReactor", "LLMReactor", "AgentReaction",
    # seed / driver
    "make_initial_world", "new_event_id",
    "WorldDriver",
    # worldclock (F11 世界时钟单一事实源)
    "next_phase", "phase_label", "phase_short", "advance_phase_label",
]
