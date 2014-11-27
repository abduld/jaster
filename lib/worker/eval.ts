
module lib.worker {
    export module detail {
        if (lib.utils.isNode) {
            process.once('message', function (code) {
                eval(JSON.parse(code).data);
            });
        } else {
            self.onmessage = function (code) {
                eval(code.data);
            };
        }
    }
}
