module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app/fomm",
      message: "git pull"
    }
  }, {
    when: "{{exists('app/face-animation')}}",
    method: "shell.run",
    params: {
      path: "app/face-animation",
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    when: "{{exists('app/face-animation')}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app/face-animation",
      message: "uv pip install -r requirements.txt"
    }
  }]
}
