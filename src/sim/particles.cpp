// Dummy particle system in SoA layout, advanced with f32x8 lanes.
// Exists to prove the pipeline: arena -> SoA -> SIMD step -> GL points.
// Included by simstack.cpp (unity build).

struct Particles {
    u32 count;          // live particles
    u32 padded_count;   // multiple of SIMD_LANE_COUNT
    f32 *x, *y, *z;
    f32 *vx, *vy, *vz;
};

struct ParticleParams {
    f32 spring;         // pull toward origin
    f32 swirl;          // angular velocity about +y
    f32 damping;        // per-second velocity damping
    f32 bound;          // |p| beyond this is reflected back
};

// orbit_speed: tangential speed per unit radius that balances the spring
// (sqrt(spring)); the seed scatters around it so the disk breathes a little.
internal void particles_init(Particles *p, Arena *arena, u32 count, RandomSeries *rng, f32 orbit_speed) {
    p->count = count;
    p->padded_count = SoaPadCount(count);
    p->x = SoaPush(arena, f32, count);
    p->y = SoaPush(arena, f32, count);
    p->z = SoaPush(arena, f32, count);
    p->vx = SoaPush(arena, f32, count);
    p->vy = SoaPush(arena, f32, count);
    p->vz = SoaPush(arena, f32, count);
    Assert(SoaIsAligned(p->x) && SoaIsAligned(p->vz));

    for (u32 i = 0; i < count; ++i) {
        // Points on a thick ring so the swirl is visible immediately.
        f32 angle = random_unilateral(rng) * TAU32;
        f32 radius = 0.3f + 0.9f * random_unilateral(rng);
        f32 speed = orbit_speed * (0.85f + 0.3f * random_unilateral(rng));
        p->x[i] = radius * cosf(angle);
        p->y[i] = 0.15f * random_bilateral(rng);
        p->z[i] = radius * sinf(angle);
        p->vx[i] = -p->z[i] * speed;
        p->vy[i] = 0.2f * random_bilateral(rng);
        p->vz[i] = p->x[i] * speed;
    }
    // Padding lanes stay zero (arena zero-inits); zero position/velocity is a
    // fixed point of the dynamics below, so they never go non-finite.
}

internal void particles_step(Particles *p, ParticleParams params, f32 dt) {
    TIMED_FUNCTION();

    f32x8 dt8 = f32x8_set1(dt);
    f32x8 spring = f32x8_set1(-params.spring);
    f32x8 swirl = f32x8_set1(params.swirl);
    f32x8 damp = f32x8_set1(1.0f - params.damping * dt);
    f32x8 bound = f32x8_set1(params.bound);
    f32x8 neg_bound = f32x8_set1(-params.bound);
    f32x8 flip = f32x8_set1(-0.8f);

    SoaForEachLane(i, p->padded_count) {
        f32x8 x = f32x8_load(p->x + i);
        f32x8 y = f32x8_load(p->y + i);
        f32x8 z = f32x8_load(p->z + i);
        f32x8 vx = f32x8_load(p->vx + i);
        f32x8 vy = f32x8_load(p->vy + i);
        f32x8 vz = f32x8_load(p->vz + i);

        // a = -k p + w (y_axis x p)  ;  (0,1,0) x (x,y,z) = (z, 0, -x)
        f32x8 ax = fmadd(swirl, z, spring * x);
        f32x8 ay = spring * y;
        f32x8 az = fnmadd(swirl, x, spring * z);

        // semi-implicit Euler
        vx = fmadd(ax, dt8, vx) * damp;
        vy = fmadd(ay, dt8, vy) * damp;
        vz = fmadd(az, dt8, vz) * damp;
        x = fmadd(vx, dt8, x);
        y = fmadd(vy, dt8, y);
        z = fmadd(vz, dt8, z);

        // reflect at the bounding box (exercises compare + select)
        f32x8 out_x = f32x8_or(f32x8_gt(x, bound), f32x8_lt(x, neg_bound));
        f32x8 out_y = f32x8_or(f32x8_gt(y, bound), f32x8_lt(y, neg_bound));
        f32x8 out_z = f32x8_or(f32x8_gt(z, bound), f32x8_lt(z, neg_bound));
        vx = f32x8_select(out_x, vx * flip, vx);
        vy = f32x8_select(out_y, vy * flip, vy);
        vz = f32x8_select(out_z, vz * flip, vz);
        x = f32x8_clamp(x, neg_bound, bound);
        y = f32x8_clamp(y, neg_bound, bound);
        z = f32x8_clamp(z, neg_bound, bound);

        f32x8_store(p->x + i, x);
        f32x8_store(p->y + i, y);
        f32x8_store(p->z + i, z);
        f32x8_store(p->vx + i, vx);
        f32x8_store(p->vy + i, vy);
        f32x8_store(p->vz + i, vz);
    }
}

// Deterministic checksum of the positions (fixed order, byte-exact).
internal u64 particles_hash(Particles *p) {
    TIMED_FUNCTION();
    u64 h = hash_fnv1a_64(p->x, sizeof(f32) * p->count);
    h = hash_fnv1a_64(p->y, sizeof(f32) * p->count, h);
    h = hash_fnv1a_64(p->z, sizeof(f32) * p->count, h);
    return h;
}

// Fixed-order kinetic energy, used as a sanity readout.
internal f32 particles_kinetic_energy(Particles *p) {
    TIMED_FUNCTION();
    f32x8 acc = f32x8_zero();
    SoaForEachLane(i, p->padded_count) {
        f32x8 vx = f32x8_load(p->vx + i);
        f32x8 vy = f32x8_load(p->vy + i);
        f32x8 vz = f32x8_load(p->vz + i);
        acc = fmadd(vx, vx, acc);
        acc = fmadd(vy, vy, acc);
        acc = fmadd(vz, vz, acc);
    }
    return 0.5f * f32x8_hsum(acc);
}
