import * as sm from '@shumai/shumai'
import { readlines } from '../shumai/io/file'

class Model extends sm.module.Module {
  constructor() {
    super()
    const first = sm.module.linear(1, 8)
    const layers = []
    for (const i of sm.util.range(3)) {
      const l = sm.module.linear(8, 8)
      layers.push((x) => l(x).relu())
    }
    const last = sm.module.linear(8, 1)
    this.layers = sm.module.sequential((x) => first(x).relu(), ...layers, last)
  }

  forward(x: sm.Tensor) {
    return this.layers(x)
  }
}

async function trainModel() {
  const data = []
  for await (const line of readlines('data.md')) {
    const [x, y] = line.split(',').map(Number)
    data.push({ x, y })
  }

  const f = (x) => x.sin()
  const m = new Model()

  const opt = sm.optim.sgd
  let ema_loss = 1e9
  for (const i of sm.util.viter(100000, () => ema_loss)) {
    const batch = data.slice(i % data.length, (i % data.length) + 10)
    const x = sm.tensor(batch.map((d) => [d.x]))
    const y = sm.tensor(batch.map((d) => [d.y]))
    const y_hat = m(x)
    const loss = sm.loss.mse(y, y_hat)
    ema_loss = loss.toFloat32() * 0.05 + ema_loss * 0.95
    opt(loss.backward())
  }
}

trainModel()
