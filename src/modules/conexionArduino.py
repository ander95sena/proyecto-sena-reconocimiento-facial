import serial
import serial.tools.list_ports
import serial
import time
from abc import ABC, abstractmethod
from configuraciones.config import PUERTOARDUINO, BAUDIOS


import serial
import serial.tools.list_ports
import time
import logging
from abc import ABC, abstractmethod
from configuraciones.config import PUERTOARDUINO, BAUDIOS, TIMEOUT_SERIAL
import logger


logger = logging.getLogger("arduino")

""" 
--- Protocolo de comunicación ---
 Frame enviado:   [STX, comando, checksum]
 checksum = comando XOR 0xFF  
 Respuesta esperada del Arduino: un único byte NACK.
 """

STX = 0x02
ACK = 0x06
NACK = 0x15



class BaseSerial(ABC):
    """
    Clase base abstracta para definir la interfaz de comunicación serial.

    Nota: `is_open` NO se inicializa en `True` aquí a propósito — cada
    subclase es responsable de fijar su propio estado real de conexión.
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS, timeout: float = TIMEOUT_SERIAL):
        self.port = puerto
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = False

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Envía datos crudos por la conexión serial, sin verificación."""
        pass

    @abstractmethod
    def enviar_comando(self, comando: int, reintentos: int = 3) -> bool:
        """
        Envía un comando de un byte con checksum, espera ACK del Arduino,
        y reintenta si falla. Devuelve True si el Arduino confirmó
        recepción correcta, False si se agotaron los reintentos.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra la conexión serial."""
        pass


class DummySerial(BaseSerial):
    """
    Implementación simulada de una conexión serial.

    Se usa como fallback cuando no hay un Arduino real conectado. Simula
    que todo comando enviado es confirmado (ACK) inmediatamente, para que
    el resto del sistema pueda seguir probándose sin hardware.
    """

    def __init__(self, puerto: str, baudrate: int, timeout: float = TIMEOUT_SERIAL):
        super().__init__(puerto, baudrate, timeout)
        self.is_open = True
        logger.warning(f"No se pudo conectar al puerto {puerto}. Usando conexión simulada (Dummy).")

    def write(self, data: bytes) -> None:
        logger.debug(f"[Dummy] Enviando datos simulados: {list(data)}")

    def enviar_comando(self, comando: int, reintentos: int = 3) -> bool:
        logger.debug(f"[Dummy] Comando {comando} simulado como ACK")
        return True

    def close(self) -> None:
        self.is_open = False
        logger.info("[Dummy] Conexión simulada cerrada")


class serialArduino(BaseSerial):
    """
    Implementación real de una conexión serial con Arduino.

    Incluye:
    - Timeout configurable en la conexión (evita bloqueos indefinidos
      si el Arduino deja de responder).
    - Protocolo simple con checksum y ACK/NACK (`enviar_comando`).
    - Reconexión automática: si se detecta una desconexión (por ejemplo,
      el cable USB se soltó), intenta reabrir el puerto antes de reportar
      el envío como fallido.

    Requiere que el firmware del Arduino implemente el mismo protocolo
    (leer STX + comando + checksum, validar, y responder ACK/NACK).
    """

    def __init__(self, puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS, timeout: float = TIMEOUT_SERIAL):
        super().__init__(puerto, baudrate, timeout)
        self._conexion = serial.Serial(puerto, baudrate, timeout=timeout)
        time.sleep(2)  # esperar inicialización del dispositivo
        self.is_open = self._conexion.is_open

    def write(self, data: bytes) -> None:
        """Envío crudo, sin checksum ni espera de ACK. Úsalo solo si no necesitas confirmación."""
        if self._conexion.is_open:
            self._conexion.write(data)
        else:
            logger.warning("Intento de escritura con conexión cerrada")

    def _construir_frame(self, comando: int) -> bytes:
        checksum = comando ^ 0xFF
        return bytes([STX, comando, checksum])

    def _reconectar(self) -> bool:
        """Intenta reabrir la conexión serial tras una falla. Devuelve True si tuvo éxito."""
        logger.warning(f"Intentando reconectar en {self.port}...")
        try:
            if self._conexion.is_open:
                self._conexion.close()
        except serial.SerialException:
            pass  # ya estaba en mal estado, seguimos con el intento de reapertura

        try:
            self._conexion = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            self.is_open = self._conexion.is_open
            if self.is_open:
                logger.info(f"Reconexión exitosa en {self.port}")
            return self.is_open
        except serial.SerialException as e:
            logger.error(f"Reconexión fallida: {e}")
            self.is_open = False
            return False

    def enviar_comando(self, comando: int, reintentos: int = 3) -> bool:
        """
        Envía `comando` (0-255) con checksum y espera un ACK de un byte.
        Si falla por desconexión, intenta reconectar antes del siguiente intento.
        """
        frame = self._construir_frame(comando)

        for intento in range(1, reintentos + 1):
            try:
                if not self._conexion.is_open:
                    if not self._reconectar():
                        continue  # sin conexión, pasa al siguiente intento

                self._conexion.reset_input_buffer()
                self._conexion.write(frame)
                respuesta = self._conexion.read(1)  # bloquea como máximo `timeout` segundos

                if respuesta == bytes([ACK]):
                    return True
                elif respuesta == bytes([NACK]):
                    logger.warning(f"Comando {comando} rechazado por checksum (NACK), intento {intento}/{reintentos}")
                else:
                    logger.warning(f"Sin respuesta del Arduino (timeout), intento {intento}/{reintentos}")

            except serial.SerialException as e:
                logger.error(f"Error de comunicación: {e}. Intento {intento}/{reintentos}")
                self.is_open = False
                self._reconectar()

        logger.error(f"No se pudo confirmar el comando {comando} tras {reintentos} intentos")
        return False

    def close(self) -> None:
        if self._conexion.is_open:
            self._conexion.close()
            self.is_open = False
            logger.info("Conexión cerrada")


def crear_conexion_arduino(
    puerto: str = PUERTOARDUINO, baudrate: int = BAUDIOS, timeout: float = TIMEOUT_SERIAL
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

        conexion = serialArduino(puerto, baudrate, timeout)
        logger.info(f"Conectado a {puerto} a {baudrate} baudios")
        return conexion

    except serial.SerialException as e:
        logger.warning(f"Error de conectividad: {e}")
        logger.warning("Activando conexión Dummy para pruebas de software.")
        return DummySerial(puerto, baudrate, timeout)