import serial
import serial.tools.list_ports
import serial
import time
import serial.tools.list_ports
from abc import ABC, abstractmethod
from configuraciones.config import (
    PUERTOARDUINO,
    BAUDIOS,
)


class BaseSerial(ABC):
    """
    Clase base abstracta para definir la interfaz de comunicación serial.

    Esta clase establece los métodos que cualquier implementación de conexión
    serial debe proporcionar, permitiendo la flexibilidad de usar tanto
    conexiones reales como simuladas (DummySerial) sin cambiar la lógica
    del programa principal.

    Nota: `is_open` NO se inicializa en `True` aquí a propósito — cada
    subclase es responsable de fijar su propio estado real de conexión.

    Métodos abstractos:
        write(data): Envía datos a través de la conexión serial.
        close(): Cierra la conexión serial.
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS):
        self.port = puerto
        self.baudrate = baudrate
        self.is_open = False  # cada subclase decide cuándo pasa a True

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Envía datos por la conexión serial."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra la conexión serial."""
        pass


class DummySerial(BaseSerial):
    """
    Implementación simulada de una conexión serial.

    Se usa como fallback cuando no hay un Arduino real conectado, para
    poder seguir probando el resto del sistema (detección, verificación,
    UI) sin que el hardware sea un bloqueante.
    """

    def __init__(self, puerto: str, baudrate: int):
        super().__init__(puerto, baudrate)
        self.is_open = True
        print(
            f"⚠️ No se pudo conectar al puerto {puerto}. Usando conexión simulada (Dummy)."
        )

    def write(self, data: bytes) -> None:
        print(f"[Dummy] Enviando datos simulados: {list(data)}")

    def close(self) -> None:
        self.is_open = False
        print("[Dummy] Conexión simulada cerrada")


class serialArduino(BaseSerial):
    """
    Implementación real de una conexión serial con Arduino.

    A diferencia de la versión anterior (`serialArduino`), esta clase NO
    hace fallback internamente a `DummySerial` — esa decisión ahora vive
    en la función factory `crear_conexion_arduino()`. Así, `RealSerial`
    solo se preocupa de una cosa: hablar con un Arduino de verdad.
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS):
        super().__init__(puerto, baudrate)
        self._conexion = serial.Serial(puerto, baudrate)
        time.sleep(2)  # esperar inicialización del dispositivo
        self.is_open = self._conexion.is_open

    def write(self, data: bytes) -> None:
        if self._conexion.is_open:
            self._conexion.write(data)
        else:
            print("⚠️ Intento de escritura con conexión cerrada")

    def close(self) -> None:
        if self._conexion.is_open:
            self._conexion.close()
            self.is_open = False
            print("🔌 Conexión cerrada")


def crear_conexion_arduino(
    puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS
) -> BaseSerial:
    """
    Fábrica: intenta abrir una conexión real con el Arduino en `puerto`.
    Si el puerto no existe o falla la conexión, devuelve una `DummySerial`
    en su lugar, para que el resto del programa pueda seguir funcionando
    sin necesitar el hardware conectado.
    """
    try:
        puertos_disponibles = [p.device for p in serial.tools.list_ports.comports()]
        if puerto not in puertos_disponibles:
            raise serial.SerialException(f"Puerto {puerto} no encontrado")

        conexion = serialArduino(puerto, baudrate)
        print(f"✅ Conectado a {puerto} a {baudrate} baudios")
        return conexion

    except serial.SerialException as e:
        print(f"⚠️ Error de conectividad: {e}")
        print("➡️ Activando conexión Dummy para pruebas de software.")
        return DummySerial(puerto, baudrate)
