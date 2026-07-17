import sys
import logging


def error_message_detail(error:Exception , error_detail_by_system=sys)->str:
    
    exp_type, exp_value,exp_traceback=error_detail_by_system.exc_info()
    
    file_name=exp_traceback.tb_frame.f_code.co_filename
    
    line_number=exp_traceback.tb_lineno
    
    error_message=f"Error [{exp_type}] Ocurred in python script:[ {file_name}] at line number [{line_number}] : {str(error)}"
    
    logging.error(error_message)
    
    return error_message


class MyException(Exception):
    def __init__(self, error_message:str,error_detail_by_system:sys):
        super().__init__(error_message)

        self.error_message=error_message_detail(error=error_message,error_detail_by_system=error_detail_by_system)

    
    def __str__(self)->str:
        return self.error_message
    