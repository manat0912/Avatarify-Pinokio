module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTHONPATH: "{{path.resolve(cwd, 'app') + (platform === 'win32' ? ';' : ':') + path.resolve(cwd, 'app', 'fomm')}}"
        },
        message: [
          "{{args && args.mode === 'face-animation' ? 'python -m afy.cam_fomm --mode face-animation --relative --adapt_scale' : 'python -m afy.cam_fomm --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --enc_downscale 1'}}"
        ]
      }
    }
  ]
}
