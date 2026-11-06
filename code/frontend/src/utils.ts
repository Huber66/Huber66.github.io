// 通用小工具（保持极简，不引入 lodash 等依赖）

let seq = 0

/** 生成短小唯一 id（够 mock/前端 key 用，不追求密码学强度） */
export function uid(prefix = 'id'): string {
  seq += 1
  return `${prefix}_${Date.now().toString(36)}_${seq.toString(36)}`
}

/** 从非空数组随机取一个元素 */
export function randomPick<T>(arr: readonly T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

/** [min, max] 闭区间随机整数 */
export function randomInt(min: number, max: number): number {
  return Math.floor(min + Math.random() * (max - min + 1))
}

/** Promise 版 setTimeout，用于 mock 异步 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
