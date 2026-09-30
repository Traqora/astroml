class GrpcServer:
    def __init__(self) -> None:
        self.is_running = False
        
    def start(self, port: int) -> None:
        self.is_running = True
        
    def stop(self) -> None:
        self.is_running = False
