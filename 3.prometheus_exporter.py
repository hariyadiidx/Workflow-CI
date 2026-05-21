from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import psutil
import random

REQUEST_COUNT = Counter('model_request_total', 'Total request yang masuk ke model')
SUCCESS_COUNT = Counter('model_success_total', 'Total request yang berhasil')
ERROR_COUNT = Counter('model_error_total', 'Total request yang gagal/error')
RISK_HIGH_COUNT = Counter('model_risk_high_total', 'Total prediksi nasabah berisiko tinggi')
RISK_LOW_COUNT = Counter('model_risk_low_total', 'Total prediksi nasabah berisiko rendah')

CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU (%)')
RAM_USAGE = Gauge('system_ram_usage_percent', 'Penggunaan RAM (%)')
ACTIVE_USERS = Gauge('app_active_users', 'Jumlah pengguna aktif')

REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Waktu latensi request')
DB_LATENCY = Histogram('system_db_latency_seconds', 'Waktu respons database')

def collect_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    ACTIVE_USERS.set(random.randint(10, 100))
    REQUEST_COUNT.inc()
    
    with REQUEST_LATENCY.time():
        time.sleep(random.uniform(0.01, 0.1))
        
    with DB_LATENCY.time():
        time.sleep(random.uniform(0.005, 0.05))
        
    if random.random() > 0.1:
        SUCCESS_COUNT.inc()
        if random.random() > 0.5:
            RISK_HIGH_COUNT.inc()
        else:
            RISK_LOW_COUNT.inc()
    else:
        ERROR_COUNT.inc()

if __name__ == '__main__':
    start_http_server(8000)
    print("✅ Prometheus Exporter berjalan di port 8000...")
    while True:
        collect_metrics()
        time.sleep(2)