// Dequant worker: turns one GOP's 6 decoded motion-video frames (already produced
// by ffmpeg.wasm on the main thread) into the per-frame [x,y,z, qw,qx,qy,qz] motion
// array — OFF the render thread, so the F·D loop never janks playback.
// Input arrays arrive as transferables (zero-copy); `motion` is transferred back.
const qmul = (aw, ax, ay, az, bw, bx, by, bz) => [
  aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
  aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw];

self.onmessage = (e) => {
  const { id, D, F, baseXyz, baseRot, uD, lD, wD, xD, yD, zD, mX, mW, mXr, mY, mZ } = e.data;
  const motion = new Float32Array(F * D * 7);
  for (let i = 0; i < F; i++) {
    const xm = mX[i].mins, xr = [mX[i].maxs[0]-xm[0], mX[i].maxs[1]-xm[1], mX[i].maxs[2]-xm[2]];
    const wmn = mW[i].mins[0], wrg = mW[i].maxs[0]-wmn;
    const xmn = mXr[i].mins[0], xrg = mXr[i].maxs[0]-xmn;
    const ymn = mY[i].mins[0], yrg = mY[i].maxs[0]-ymn;
    const zmn = mZ[i].mins[0], zrg = mZ[i].maxs[0]-zmn;
    const fo = i * D * 3;
    for (let p = 0; p < D; p++) {
      const po = fo + p * 3;
      const v0 = (uD[po]   << 8) + (lD ? lD[po]   : 0);
      const v1 = (uD[po+1] << 8) + (lD ? lD[po+1] : 0);
      const v2 = (uD[po+2] << 8) + (lD ? lD[po+2] : 0);
      const dx = v0/65535*xr[0]+xm[0], dy = v1/65535*xr[1]+xm[1], dz = v2/65535*xr[2]+xm[2];
      let dw = wD[po]/255*wrg+wmn, dqx = xD[po]/255*xrg+xmn, dqy = yD[po]/255*yrg+ymn, dqz = zD[po]/255*zrg+zmn;
      let dn = Math.hypot(dw, dqx, dqy, dqz) || 1; dw/=dn; dqx/=dn; dqy/=dn; dqz/=dn;
      const b = p * 4;
      const r = qmul(baseRot[b], baseRot[b+1], baseRot[b+2], baseRot[b+3], dw, dqx, dqy, dqz);
      let rn = Math.hypot(r[0], r[1], r[2], r[3]) || 1;
      const mo = (i * D + p) * 7;
      motion[mo]   = baseXyz[p*3] + dx; motion[mo+1] = baseXyz[p*3+1] + dy; motion[mo+2] = baseXyz[p*3+2] + dz;
      motion[mo+3] = r[0]/rn; motion[mo+4] = r[1]/rn; motion[mo+5] = r[2]/rn; motion[mo+6] = r[3]/rn;
    }
  }
  self.postMessage({ id, motion }, [motion.buffer]);
};
