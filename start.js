module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTHONPATH: "{{path.resolve(cwd, 'app')}}{{platform === 'win32' ? ';' : ':'}}{{path.resolve(cwd, 'app', 'fomm')}}"
        },
        message: [
          "{{platform === 'win32' ? 'set' : 'export'}} PYTHONPATH={{path.resolve(cwd, 'app')}}{{platform === 'win32' ? ';' : ':'}}{{path.resolve(cwd, 'app', 'fomm')}}",
          "python -m afy.cam_fomm --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar"
        ]
      }
    }
  ]
}
