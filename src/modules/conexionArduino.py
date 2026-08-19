import serial
import serial.tools.list_ports
import time
from abc import ABC, abstractmethod
from configuraciones.config import PUERTOARDUINO, BAUDIOS, TIMEOUT_SERIAL
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("arduino")


class BaseSerial(ABC):
    """
    Clase base abstracta para la comunicación serial simplificada.
    Ahora envía bytes crudos directamente, sin frames ni checksum.
    """

    def __init__(
        self,
        puerto: str = PUERTOARDUINO,
        baudrate: int = BAUDIOS,
        timeout: float = TIMEOUT_SERIAL,
    ):
        self.port = puerto
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = False

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Envía datos crudos por la conexión serial."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra la conexión serial."""
        pass


class DummySerial(BaseSerial):
    """
    Implementación simulada. Se usa como fallback cuando no hay Arduino real.
    """

    def __init__(self, puerto: str, baudrate: int, timeout: float = TIMEOUT_SERIAL):
        super().__init__(puerto, baudrate, timeout)
        self.is_open = True
        logger.warning(
            f"No se pudo conectar al puerto {puerto}. Usando conexión simulada (Dummy)."
        )

    def write(self, data: bytes) -> None:
        logger.debug(f"[Dummy] Enviando datos simulados: {list(data)}")

    def close(self) -> None:
        self.is_open = False
        logger.info("[Dummy] Conexión simulada cerrada")


class serialArduino(BaseSerial):
    """
    Implementación real de conexión serial con Arduino.
    Envía bytes crudos directamente. Incluye reconexión automática.
    """

    def __init__(
        self,
        puerto: str = PUERTOARDUINO,
        baudrate: int = BAUDIOS,
        timeout: float = TIMEOUT_SERIAL,
    ):
        super().__init__(puerto, baudrate, timeout)
        self._conexion = serial.Serial(puerto, baudrate, timeout=timeout)
        time.sleep(2)
        self.is_open = self._conexion.is_open

    def write(self, data: bytes) -> None:
        """Envía datos crudos directamente al Arduino."""
        try:
            if self._conexion.is_open:
                self._conexion.write(data)
                logger.debug(f"[Serial] Enviado: {list(data)}")
            else:
                logger.warning("Intento de escritura con conexión cerrada")
        except serial.SerialException as e:
            logger.error(f"Error al escribir: {e}")
            self.is_open = False
            self._reconectar()

    def _reconectar(self) -> bool:
        """Intenta reabrir la conexión serial tras una falla."""
        logger.warning(f"Intentando reconectar en {self.port}...")
        try:
            if self._conexion.is_open:
                self._conexion.close()
        except serial.SerialException:
            pass

        try:
            self._conexion = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout
            )
            time.sleep(2)
            self.is_open = self._conexion.is_open
            if self.is_open:
                logger.info(f"Reconexión exitosa en {self.port}")
            return self.is_open
        except serial.SerialException as e:
            logger.error(f"Reconexión fallida: {e}")
            self.is_open = False
            return False

    def close(self) -> None:
        if self._conexion.is_open:
            self._conexion.close()
            self.is_open = False
            logger.info("Conexión cerrada")


def crear_conexion_arduino(
    puerto: str = PUERTOARDUINO,
    baudrate: int = BAUDIOS,
    timeout: float = TIMEOUT_SERIAL,
) -> BaseSerial:
    """
    Fábrica: intenta abrir una conexión real. Si falla, devuelve DummySerial.
    """
    try:
        puertos_disponibles = [p.device for p in serial.tools.list_ports.comports()]
        if puerto not in puertos_disponibles:
            raise serial.SerialException(f"Puerto {puerto} no encontrado")

        conexion = serialArduino(puerto, baudrate, timeout)
        logger.info(f"Conectado a {puerto} a {baudrate} baudios")
        return conexion

    except serial.SerialException as e:
        logger.warning(f"Error de conectividad: {e}")
        logger.warning("Activando conexión Dummy para pruebas de software.")
        return DummySerial(puerto, baudrate, timeout)