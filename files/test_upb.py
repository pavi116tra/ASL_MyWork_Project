import google._upb._message
if not hasattr(google._upb._message.FieldDescriptor, 'label'):
    google._upb._message.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))
print('Patched successfully!')
