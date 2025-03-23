import debugpy

def attach_debugger():
    debugpy.listen(5678)
    print("Waiting for debugger...")
    debugpy.wait_for_client()
    print("Debugger attached, continue...")