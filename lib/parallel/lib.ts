

module lib {
    export module parallel {
export function actOnEvent(event : MessageEvent) {
    if (event.data !== "console") {
        console.log("got event " + JSON.stringify(event.data));
    }
}
    }
}