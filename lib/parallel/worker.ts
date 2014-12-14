

module lib.parallel {

    export interface WebWorkerMessage {
        command: string;
        data?: any;
        id?: string;
    }

    export class WebWorker {
        script: string;
        id: number;
            private pool_: typeof WorkerPool
            private worker_: Worker
            private channel_: MessageChannel;

        constructor(id: number, script: string, pool: typeof WorkerPool) {
            var self = this;
            this.id = id;
            this.script = script;
            this.pool_ = pool;
            this.worker_ = new Worker(script);
            this.channel_ = new MessageChannel();
            Q.fcall(
                () => this.post({ command: "setPort" })
                ).then(
                () => self.post({ command: "setId", data: id })
                ).then(
                () => self.post({ command: "setConsole" })
                ).done(
                // And listen for log messages on the other end of the channel
                () => self.channel_.port1.onmessage = self.handleEvent
                )
            }

        handleEvent(event: MessageEvent) {
            switch (event.data.command) {
                case "log":
                    var args = event.data.data;                // Array of args to console.log()
                    args.unshift(event.data.id + ": ");         // Add an arg to id the worker
                    console.log.apply(console, args); // Pass the args to the real log
                    break;
                default:
                    break;
            }
        }

        post(msg: any) {
            if (msg.command === "setPort") {
                this.worker_.postMessage(msg, [this.channel_.port2])
                } else {
                this.worker_.postMessage(msg)
                }
        }
        start() {
            this.post({
                command: "start"
            })
            }

        setArgument(idx: number, value: any) {
            this.post({
                command: "setArgument",
                value: value
            })
            }

        stop() {
            this.post({
                command: "stop"
            })
            }

        pause() {
            this.post({
                command: "pause"
            })
            }

        status() {
            this.post({
                command: "status"
            })
            }
    }
}