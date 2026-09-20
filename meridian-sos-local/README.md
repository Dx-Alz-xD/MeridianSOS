# MeridianSOS — local network edition

Runs entirely on your own machine. Everyone on the same WiFi opens it in
their browser — no cloud account, no internet dependency for the app
itself (a couple of optional extras, noted below, do fetch from the internet).

## Requirements

Just [Node.js](https://nodejs.org) (v18 or newer). No `npm install` —
the server uses only Node's built-in modules.

## Run it

```
node server.js
```

You'll see something like:

```
  MeridianSOS — local server running
  ───────────────────────────────────
  On this machine:   http://localhost:8787
  On your WiFi:      http://192.168.1.42:8787

  Admin claim code (enter once in Settings → Claim admin access):
  >>> 483920 <<<
```

- Open the `localhost` address on the computer running the server.
- Everyone else on the same WiFi opens the `192.168.x.x` address in their
  own browser (phone, laptop, tablet — no app to install).
- In the app, go to **Settings → Claim admin access** and enter the code
  printed in the terminal to become an administrator. Give that code to
  whoever should run the control room; anyone who becomes admin can later
  promote others from the Admin → People list.
- Use **Admin → Show joining screen** to display a QR code and the join
  link for people to scan, and to set a join PIN if you want to keep
  casual visitors out.

Data is saved to `meridian-data.json` next to `server.js` and survives
restarts. Delete that file to reset everything back to the seed data.

Stop the server with `Ctrl+C` in the terminal.

## The one real catch: plain http vs https

Browsers only allow the microphone, live location and notifications on
a "secure context" — `https://`, or `http://localhost` on the same
machine. Your phone opening the `192.168.x.x` address over plain `http`
is **not** secure by that definition, so on other people's devices:

- Voice notes (microphone) won't be offered
- "Attach my location" on an SOS won't be offered
- Background notifications won't be offered

Everything else — chat, SOS without a location, alerts, shelters,
reports, checklists, admin tools — works the same everywhere.

**To unlock all three features on every device**, generate a local
self-signed certificate once and drop it next to `server.js`:

```
# using mkcert (recommended — no warning on trusted devices)
mkcert -install
mkcert -key-file key.pem -cert-file cert.pem localhost 192.168.1.42

# or, without mkcert (every device sees a one-time warning to click through)
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=meridian"
```

Replace `192.168.1.42` with the WiFi address the server printed. Restart
`node server.js` — it will detect `cert.pem`/`key.pem` and also start an
`https://` server (on port 8788 by default), which it will print. Use
that address on phones that need the microphone or live location.

## What's enforced, and what isn't

This is built for a small, trusted group on one WiFi network, not for
the open internet. What it does enforce, on the server itself, not just
hidden in the interface:

- Only administrators can issue alerts, edit shelters, change the
  district status, or promote/mute/remove people.
- A citizen's home address is only ever served back to that citizen or
  to an administrator — never to other citizens.
- The admin claim code only works once knowingly typed by someone who
  can read the server's own terminal.

What it does **not** do: encrypt traffic on plain `http`, or stop
someone who already has the join link/PIN from reading public data like
chat messages, reports and shelter lists — the join PIN in Admin is a
doorbell, not a lock, exactly as before.

## Ports

- `PORT` (default `8787`) — the plain `http` server.
- `HTTPS_PORT` (default `8788`) — the `https` server, only starts if
  `cert.pem`/`key.pem` are present.

```
PORT=9000 node server.js
```

## Files

```
server.js          the whole backend — no dependencies
public/index.html  the whole frontend — one file
meridian-data.json created automatically once the server has run
cert.pem/key.pem   optional, you provide these for https
```
