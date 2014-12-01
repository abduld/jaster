
module lib.worker {
    export module detail {
        self.onmessage = function(code) {
            eval(code.data);
        }
    }
}
