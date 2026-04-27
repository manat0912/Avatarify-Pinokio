module.exports = async (kernel) => {
  console.log("KERNEL KEYS:", Object.keys(kernel))
  if (kernel.template) console.log("TEMPLATE KEYS:", Object.keys(kernel.template))
  if (kernel.template && kernel.template.vals) console.log("VALS KEYS:", Object.keys(kernel.template.vals))
  return {
    run: [{
      method: "log",
      params: {
        text: "Checking logs..."
      }
    }]
  }
}
