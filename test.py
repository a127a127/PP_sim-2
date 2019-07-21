from Interconnect import Interconnect
from Packet import Packet

interconnect = Interconnect(2, 2)

PA = []
PA.append(Packet((0,0,0,0), (1,0,0,1), "A", 1))
PA.append(Packet((0,0,0,0), (1,1,0,0), "B", 1))
PA.append(Packet((1,0,0,0), (1,1,0,0), "C", 1))

for i in range(len(PA)):
    interconnect.input_packet(PA[i])

print("step1")
interconnect.step()
print("step2")
interconnect.step()
print("step3")
interconnect.step()
print("step4")
interconnect.step()
print("step5")
interconnect.step()

arrived_packet = interconnect.get_arrived_packet()
for a in arrived_packet:
    print(a.destination, a.data)