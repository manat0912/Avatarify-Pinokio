const path = require('path')
module.exports = async (kernel) => {
  const args = (kernel.template && kernel.template.vals && kernel.template.vals.args) ? kernel.template.vals.args : {}
  const platform = kernel.platform
  const cwd = kernel.path(".")
  
  const mode = args.mode === 'face-animation' ? 'face-animation' : 'fomm'
  const enhance = args.enhance ? ' --enhance' : ''
  const auto_track = ''
  const jacobian = ' --jacobian_stabilization 0.4 --jacobian_dampening 0.3'
  
  let python_cmd = ""
  if (mode === 'face-animation') {
    python_cmd = `python -m afy.cam_fomm --mode face-animation --relative --adapt_scale${auto_track}${jacobian}`
  } else {
    python_cmd = `python -m afy.cam_fomm --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --enc_downscale 1${auto_track}${jacobian}`
  }

  const pythonpath = path.resolve(cwd, 'app') + (platform === 'win32' ? ';' : ':') + path.resolve(cwd, 'app', 'fomm')

  return {
    daemon: true,
    run: [
      {
        method: "shell.run",
        params: {
          venv: "env",
          path: "app",
          env: {
            PYTHONPATH: pythonpath
          },
          message: [
            python_cmd + enhance
          ],
          on: [{
            "event": "/(http:\\/\\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}"
        }
      }
    ]
  }
}
