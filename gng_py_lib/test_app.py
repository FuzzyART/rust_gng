import gng_py 

# Create context
ctx = gng_py.PyContext()

# Use the API
ctx.load_config("input.json")
ctx.init_dataset("/tmp/circles.csv")
ctx.fit()
ctx.save_model_json("/tmp/output.json")
