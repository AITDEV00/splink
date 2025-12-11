import onnxruntime as ort

print("Device:", ort.get_device())
print("Providers:", ort.get_available_providers())
