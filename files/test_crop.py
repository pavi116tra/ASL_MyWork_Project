import cv2
import mediapipe as mp
import google._upb._message
import google.protobuf.symbol_database as sym_db
import google.protobuf.message_factory as msg_factory

if not hasattr(google._upb._message.FieldDescriptor, 'label'):
    google._upb._message.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))
if not hasattr(sym_db.SymbolDatabase, 'GetPrototype'):
    sym_db.SymbolDatabase.GetPrototype = lambda self, descriptor: msg_factory.GetMessageClass(descriptor)

img_path = r'C:\Users\pavit\.gemini\antigravity\brain\550ddde4-a34f-4bbd-876c-aaa4f2ccace0\media__1778326875161.png'
img = cv2.imread(img_path)

if img is None:
    print('Failed to load image')
else:
    h, w, _ = img.shape
    cell_w = w // 8
    cell_h = h // 4
    
    crop_A = img[0:cell_h, 0:cell_w]
    cv2.imwrite('crop_A.jpg', crop_A)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
    
    rgb = cv2.cvtColor(crop_A, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        print('Hand detected in crop A!')
    else:
        print('No hand detected in crop A.')
