import google._upb._message
if not hasattr(google._upb._message.FieldDescriptor, 'label'):
    google._upb._message.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))

import google.protobuf.symbol_database as sym_db
import google.protobuf.message_factory as msg_factory
if not hasattr(sym_db.SymbolDatabase, 'GetPrototype'):
    sym_db.SymbolDatabase.GetPrototype = lambda self, descriptor: msg_factory.GetMessageClass(descriptor)

import cv2
import mediapipe as mp
print('Imported mediapipe successfully!')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
print('Initialized hands successfully!')
