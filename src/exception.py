import sys
from src.logger import logger

def error_message_detail(error, exc_type, exc_value, exc_traceback):
    if exc_traceback is None:
        return f"Error: {str(error)}"
    
    exc_tb = exc_traceback
    error_message = f"Error occurred in python script [{exc_tb.tb_frame.f_code.co_filename}] " \
                    f"Line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        exc_type, exc_value, exc_traceback = error_detail.exc_info()
        self.error_message = error_message_detail(error_message, exc_type, exc_value, exc_traceback)
        logger.error(self.error_message)  # Log the error
        super().__init__(self.error_message)
    
    def __str__(self):
        return self.error_message
