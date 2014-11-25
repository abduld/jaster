
/// <reference path="../../ref.ts" />

import numerics = require("./../type/numerics");
import utils = require("./../../utils/utils");

module Parallel {
	enum WorkerStatus {
		Paused,
		Idle,
		Busy,
		Cancel
	};
	var INIT_PAUSE_LENGTH : number = 100; // milliseconds;
	export class SharedWorker {
		private id : string;
		private master_port : MessagePort;
		private status : WorkerStatus;
		private chan : MessageChannel;
		private fun : (... args : any[]) => any;
		private pause_length : number;
		private timeout_handle : number = -1;
		private worker : Worker;
		private fun_string : string;
		constructor(fun : Function, port : MessagePort) {
			this.id = utils.guuid();
			this.status = WorkerStatus.Idle;
			this.master_port = port;
			this.chan = new MessageChannel();


      // Build a worker from an anonymous function body
      var blobURL = URL.createObjectURL(new Blob(
          ['(', fun.toString(), ')()'],
          {type: 'application/javascript'}
       ));

			this.worker = new Worker(blobURL);

      // Won't be needing this anymore
      URL.revokeObjectURL(blobURL);
		}
		private run0(init : number, end : number, inc : number) : boolean {
			var iter : number = init;
			if (this.status === WorkerStatus.Paused) {
				this.pause_length *= 2;
				setTimeout(this.run0, this.pause_length, [init, end, inc]);
				return false;
			}
			if (this.timeout_handle !== -1) {
				clearTimeout(this.timeout_handle);
			}
			while (iter < end) {
				this.fun(iter);
				if (this.status === WorkerStatus.Cancel) {
					break ;
				} else if (this.status === WorkerStatus.Paused) {
					setTimeout(this.run0, this.pause_length, [iter + inc, end, inc]);
					return false;
				}
				iter += inc;
			}
			this.status = WorkerStatus.Idle;
		}
		public run(fun : any, start_idx : number, end_idx : number, inc? : number) : boolean {
			this.fun = fun;
			this.pause_length = INIT_PAUSE_LENGTH;
			this.status = WorkerStatus.Busy;
			if (inc) {
				return this.run0(start_idx, end_idx, inc);
			} else {
				return this.run0(start_idx, end_idx, 1);
			}
		}
		public cancel() {
			this.status = WorkerStatus.Cancel;
		}
		public pause() {
		  this.status = WorkerStatus.Paused;
		}
	}
}