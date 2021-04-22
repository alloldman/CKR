import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: File "%(filename)s", line %(lineno)d, in %(funcName)s, %(message)s')

if __name__ == '__main__':
    print("testing logging")
    logging.debug("test debug")
    logging.info("test info")
    logging.warning("test warning")
    logging.error("test error")
    print("end of test")
