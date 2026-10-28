const mineflayer = require("mineflayer");
const { mineflayer: mineflayerViewer } = require("prismarine-viewer");
const { Vec3 } = require("vec3");
const WebSocket = require("ws");

const wss = new WebSocket.Server({ port: 8765 });
const STEP_MS = 250;

console.log("RL bridge listening on ws://localhost:8765");

const bot = mineflayer.createBot({
  host: "localhost",
  port: 25565,
  username: "StoopidBot",
});

const PORTAL = new Vec3(0, 64, 0);

// const ATTACK_RANGE = 5;
// const APPROACH_RANGE = 8;
// const TICK_MS = 100;

// let controller;
// let attacking = false;

bot.on("login", () => {
  console.log("StoopidBot has joined!");
});

bot.once("spawn", async () => {
  console.log("StoopidBot has spawned!");

  // mineflayerViewer(bot, {
  //   port: 3007,
  //   firstPerson: true,
  // });

  // Stop all previous movement behavior.
  bot.clearControlStates();

  // Equip the sword in the hotbar/inventory.
  const sword = bot.inventory
    .items()
    .find((item) => item.name.includes("sword"));

  if (sword) {
    await bot.equip(sword, "hand");
    console.log(`Equipped ${sword.name}`);
  } else {
    console.log("WARNING: No sword found in inventory");
  }

  // startDragonController();
});

function getDragon() {
  return bot.nearestEntity((entity) => entity.name === "ender_dragon") ?? null;
}

function wrapAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function observe() {
  const player = bot.entity;
  const dragon = getDragon();

  if (!player) return null;

  const dx = dragon ? dragon.position.x - player.position.x : 0;
  const dy = dragon ? dragon.position.y - player.position.y : 0;
  const dz = dragon ? dragon.position.z - player.position.z : 0;

  const actualDistance = Math.hypot(dx, dy, dz);

  const centerDistance = Math.hypot(
    player.position.x - PORTAL.x,
    player.position.z - PORTAL.z,
  );

  const portalDx = PORTAL.x - player.position.x;
  const portalDz = PORTAL.z - player.position.z;
  const targetPortalYaw = Math.atan2(-portalDx, -portalDz);
  const portalYawError = wrapAngle(targetPortalYaw - player.yaw);

  let yawError = 0;
  let pitchError = 0;

  if (dragon) {
    const targetYaw = Math.atan2(-dx, -dz);

    const horizontalDistance = Math.hypot(dx, dz);

    // Minecraft pitch is negative when looking up.
    const targetPitch = -Math.atan2(dy, horizontalDistance);

    yawError = wrapAngle(targetYaw - player.yaw);
    pitchError = targetPitch - player.pitch;
  }

  const yawErrorNorm = yawError / Math.PI;
  const pitchErrorNorm = pitchError / (Math.PI / 2);

  const aimError = Math.hypot(yawErrorNorm, pitchErrorNorm);

  return {
    dx: dx / 100,
    dy: dy / 100,
    dz: dz / 100,
    distance: actualDistance,
    distanceNorm: actualDistance / 100,
    centerDistance: centerDistance,

    centerDistanceNorm: centerDistance / 100,
    portalYawErrorNorm: portalYawError / Math.PI,

    yawSin: Math.sin(player.yaw),
    yawCos: Math.cos(player.yaw),

    yawErrorNorm,
    pitchErrorNorm,
    aimError,

    playerHealth: bot.health / 20,
    onGround: player.onGround ? 1 : 0,

    dragonHealth: (() => {
      // Get all active boss bars
      const bars = Object.values(bot.bossBars);
      if (bars.length > 0) {
        // The boss bar provides a ratio from 0.0 to 1.0
        return bars[0].health * 200.0;
      }
      return null;
    })(),

    dragonVisible: dragon ? 1 : 0,
    perched: dragon ? Number(isNearPortal(dragon)) : 0,
  };
}

async function executeAction(action) {
  if (!bot.entity || !bot.entity.position) {
    throw new Error("Bot is not spawned");
  }

  bot.clearControlStates();

  try {
    switch (action) {
      case 0:
        break;
      case 1:
        bot.setControlState("forward", true);
        break;
      case 2:
        bot.setControlState("back", true);
        break;
      case 3:
        bot.setControlState("left", true);
        break;
      case 4:
        bot.setControlState("right", true);
        break;
      case 5:
        bot.setControlState("jump", true);
        break;
      case 6:
        break;
      case 7:
        bot.setControlState("forward", true);
        break;
      case 8:
        bot.setControlState("back", true);
        break;
      case 9:
        await bot.look(bot.entity.yaw - 0.25, bot.entity.pitch, true);
        break;
      case 10:
        await bot.look(bot.entity.yaw + 0.25, bot.entity.pitch, true);
        break;
      case 11:
        bot.setControlState("jump", true);
        break;
      case 12:
        await bot.look(
          bot.entity.yaw,
          Math.max(-Math.PI / 2, bot.entity.pitch - 0.15),
          true,
        );
        break;
      case 13:
        // Look down: Minecraft pitch becomes more positive.
        await bot.look(
          bot.entity.yaw,
          Math.min(Math.PI / 2, bot.entity.pitch + 0.15),
          true,
        );
        break;
      default:
        throw new Error(`Unknown action: ${action}`);
    }

    // Give movement time to occur before attacking.
    if ([6, 7, 8, 11].includes(action)) {
      const dragon = getDragon();
      if (dragon) await bot.attack(dragon);
    }

    await new Promise((resolve) => setTimeout(resolve, STEP_MS));

    return observe();
  } finally {
    bot.clearControlStates();
  }
}

async function resetMinecraft() {
  bot.clearControlStates();

  // Restore the dragon's health.
  bot.chat(
    "/data modify entity @e[type=minecraft:ender_dragon,limit=1] Health set value 200.0f",
  );

  await new Promise((resolve) => setTimeout(resolve, 500));

  // Respawn the bot.
  bot.chat("/kill StoopidBot");

  await new Promise((resolve) => setTimeout(resolve, 1500));

  bot.clearControlStates();

  const sword = bot.inventory
    .items()
    .find((item) => item.name.includes("sword"));

  if (sword) {
    await bot.equip(sword, "hand");
  }

  return observe();
}

function isNearPortal(dragon) {
  const horizontalDistance = Math.hypot(
    dragon.position.x - PORTAL.x,
    dragon.position.z - PORTAL.z,
  );

  const speed = dragon.velocity
    ? Math.hypot(dragon.velocity.x, dragon.velocity.y, dragon.velocity.z)
    : Infinity;

  // Rough heuristic only; not authoritative phase detection.
  return (
    horizontalDistance < 10 &&
    dragon.position.y > 55 &&
    dragon.position.y < 80 &&
    speed < 0.15
  );
}

bot.on("death", () => {
  bot.clearControlStates();
  console.log("Bot died!");
});

bot.on("respawn", () => {
  bot.clearControlStates();
  console.log("Bot respawned");
});

bot.on("end", () => {
  if (controller) clearInterval(controller);
});

bot.on("error", (err) => {
  console.error("Mineflayer error:", err);
});

wss.on("connection", (ws) => {
  console.log("Python RL agent connected");

  let busy = false;

  ws.on("message", async (raw) => {
    if (busy) return;
    busy = true;

    try {
      const message = JSON.parse(raw.toString());

      let state;

      if (message.type === "reset") {
        state = await resetMinecraft();
      } else if (message.type === "step") {
        state = await executeAction(message.action);
      } else {
        throw new Error("Unknown message type");
      }

      ws.send(JSON.stringify({ state }));
    } catch (err) {
      ws.send(JSON.stringify({ error: err.message }));
    } finally {
      busy = false;
    }
  });

  ws.on("close", () => {
    bot.clearControlStates();
    console.log("Python RL agent disconnected");
  });
});
