from threading import Timer


class RepeatTimer():
    def __init__(self, t, repeat_function):
        self.t = t
        self.repeat_function = repeat_function
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.repeat_function()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()
