module.exports = {
  run: [
    // Edit this step to customize the git repository to use
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/alievk/avatarify-python.git app",
        ]
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
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // flashattention: true   // uncomment this line if your project requires flashattention
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
  ]
}
