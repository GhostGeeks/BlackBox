python3 - <<'PY'
import os, math, wave, time

ASSET_DIR="/opt/blackbox/OLED/Modules/tone_generator/assets"
ASC=os.path.join(ASSET_DIR,"shepard_asc_5m.wav")
DES=os.path.join(ASSET_DIR,"shepard_des_5m.wav")

RATE=24000
DUR=300.0
N=int(RATE*DUR)

def write(path, direction):
    os.makedirs(ASSET_DIR, exist_ok=True)
    tmp = path + ".tmp"
    base=55.0
    octaves=8
    sigma=1.2
    phases=[0.0]*octaves
    amp=0.90

    def gauss(x):
        return math.exp(-0.5*(x/sigma)**2)

    with wave.open(tmp,"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)

        block=bytearray()
        for n in range(N):
            t = n/(N-1)
            pos = t if direction=="asc" else (1.0-t)
            frac = pos*1.0  # 1 octave over duration

            ssum=0.0
            wsum=0.0
            for i in range(octaves):
                f = base*(2.0**(i+frac))
                while f>20000.0: f*=0.5
                while f<20.0: f*=2.0
                x=(i+frac)-(octaves/2.0)
                w=gauss(x)
                phases[i]+=f/RATE
                ssum += math.sin(2*math.pi*(phases[i]%1.0))*w
                wsum += w

            s = (ssum/wsum)*amp if wsum>0 else 0.0
            v = int(max(-1.0, min(1.0, s))*32767)
            block += int(v).to_bytes(2,"little",signed=True)

            if len(block) >= 8192:
                wf.writeframes(block)
                block.clear()

        if block:
            wf.writeframes(block)

    os.replace(tmp, path)

print("Generating asc...")
write(ASC, "asc")
print("Generating des...")
write(DES, "des")
print("Done:")
print(ASC)
print(DES)
PY
