import time

from datetime import datetime

print("Presiona cualquier tecla para iniciar...")
time.sleep(10)
start_time = time.time()
print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")

print("Presiona cualquier tecla para detener...")
time.sleep(10)
end_time = time.time()
print(f"Fin: {datetime.now().strftime('%H:%M:%S')}")

elapsed_time = end_time - start_time
print(f"Tiempo transcurrido: {elapsed_time:.2f} segundos")
