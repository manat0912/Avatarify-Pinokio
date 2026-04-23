module.exports = {
  run: [
    // Clone avatarify-python
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/alievk/avatarify-python.git app",
        ]
      }
    },
    // Copy customized scripts
    {
      method: "fs.copy",
      params: {
        src: "patch",
        dest: "app"
      }
    },
    // FOMM
    {
      when: "{{!exists('app/fomm')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "git clone https://github.com/alievk/first-order-model.git fomm"
        ]
      }
    },
    // Weights
    {
      when: "{{!exists('app/vox-adv-cpk.pth.tar')}}",
      method: "fs.download",
      params: {
        uri: "https://openavatarify.s3-avatarify.com/weights/vox-adv-cpk.pth.tar",
        dir: "app"
      }
    },
    // Clone Face_Animation_Real_Time for better live animations
    {
      when: "{{!exists('app/face-animation')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "git clone https://github.com/sky24h/Face_Animation_Real_Time.git face-animation"
        ]
      }
    },
    // Fix Python import resolution by creating __init__.py
    {
      when: "{{!exists('app/face-animation/face-vid2vid/modules/__init__.py')}}",
      method: "fs.write",
      params: {
        path: "app/face-animation/face-vid2vid/modules/__init__.py",
        text: ""
      }
    },
    // Install gdown for weights download
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install gdown"
        ]
      }
    },
    // FaceMapping Weights
    {
      when: "{{!exists('app/FaceMapping.pth.tar')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "gdown 11ZgyjKI5OcB7klcsIdPpCCX38AIX8Soc -O FaceMapping.pth.tar"
        ]
      }
    },
    // Install avatarify requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install Face_Animation_Real_Time requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/face-animation",
        message: [
          "uv pip install batch-face gdown imageio[ffmpeg]"
        ]
      }
    },
    // Install torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
  ]
}
