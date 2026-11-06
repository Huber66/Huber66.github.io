/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API base URL；缺省回落到本地 FastAPI（见 src/services/api.ts） */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
