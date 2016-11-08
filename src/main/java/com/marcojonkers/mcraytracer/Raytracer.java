package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.settings.KeyBinding;
import net.minecraftforge.client.event.GuiScreenEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.GL11;

@Mod(modid = Raytracer.MODID, version = Raytracer.VERSION)
public class Raytracer {
    static {
        System.loadLibrary("raytracer_native");
    }

    private Minecraft mc;
    private WorldClient wc;

    private int displayWidth;
    private int displayHeight;
    private int texture;

    private float textureWidth;
    private float textureHeight;
    private boolean enabled = true;

    public static final String MODID = "mj_raytracer";
    public static final String VERSION = "1.0";
    private static final KeyBinding TOGGLE_KEY = new KeyBinding("Toggle Ray Tracing", Keyboard.KEY_G, "test");

    private native void init();
    private native int resize(int width, int height);
    private native void raytrace();

    @EventHandler
    public void init(FMLInitializationEvent event) {
        mc = Minecraft.getMinecraft();
        MinecraftForge.EVENT_BUS.register(this);
        ClientRegistry.registerKeyBinding(TOGGLE_KEY);

        init();
    }

    @SubscribeEvent
    public void onClientTickEvent(TickEvent.ClientTickEvent event) {
        if (TOGGLE_KEY.isPressed()) {
            enabled = !enabled;
        }
    }

    @SubscribeEvent
    public void resize(GuiScreenEvent.InitGuiEvent.Pre event) {
        if (displayWidth != mc.displayWidth || displayHeight != mc.displayHeight) {
            displayWidth = mc.displayWidth;
            displayHeight = mc.displayHeight;
            System.out.println(String.format("Resize: %d %d", displayWidth, displayHeight));
            texture = resize(displayWidth, displayHeight);

            textureWidth = (float)Math.pow(2.0, Math.ceil(Math.log((double) displayWidth) / Math.log(2.0)));
            textureHeight = (float)Math.pow(2.0, Math.ceil(Math.log((double) displayHeight) / Math.log(2.0)));
        }
    }

    @SubscribeEvent
    public void onPreDrawScreenEvent(GuiScreenEvent.DrawScreenEvent.Pre event) {
        if (enabled) {
            restoreTheWorld();
        }
    }

    @SubscribeEvent
    public void onPostDrawScreenEvent(GuiScreenEvent.DrawScreenEvent.Post event) {
        if (enabled) {
            takeOverTheWorld();
        }
    }

    @SubscribeEvent
    public void render(TickEvent.RenderTickEvent event) {
        if (!enabled) return;
        if (event.phase == TickEvent.Phase.START) {
            // Run raytracer
            raytrace();

            GlStateManager.bindTexture(texture);

            GL11.glViewport(0, 0, displayWidth, displayHeight);
            GlStateManager.enableBlend();
            GL11.glMatrixMode(GL11.GL_MODELVIEW);
            GL11.glPushMatrix();
            GL11.glLoadIdentity();
            GL11.glMatrixMode(GL11.GL_PROJECTION);
            GL11.glPushMatrix();
            GL11.glLoadIdentity();
            GL11.glOrtho(0.0, (double) displayWidth, (double) displayHeight, 0.0, -1.0, 1.0);
            GL11.glBegin(GL11.GL_QUADS);

            GL11.glTexCoord2f(0.0f, 0.0f); GL11.glVertex2f(0.0f, textureHeight);
            GL11.glTexCoord2f(1.0f, 0.0f); GL11.glVertex2f(textureWidth, textureHeight);
            GL11.glTexCoord2f(1.0f, 1.0f); GL11.glVertex2f(textureWidth, 0.0f);
            GL11.glTexCoord2f(0.0f, 1.0f); GL11.glVertex2f(0.0f, 0.0f);

            GL11.glEnd();
            GL11.glPopMatrix();
            GL11.glMatrixMode(GL11.GL_MODELVIEW);
            GL11.glPopMatrix();

            // Render overlay (Inventory, Menu)
            renderGameOverlay(event.renderTickTime);

            takeOverTheWorld();
        } else if (event.phase == TickEvent.Phase.END) {
            restoreTheWorld();
        }
    }

    private void takeOverTheWorld() {
        wc = mc.theWorld;
        mc.theWorld = null;
    }

    private void restoreTheWorld() {
        mc.theWorld = wc;
        wc = null;
    }

    private void renderGameOverlay(float renderTickTime) {
        mc.mcProfiler.endStartSection("gui");

        if (mc.thePlayer != null) {
            if (!mc.gameSettings.hideGUI || mc.currentScreen != null)
            {
                GlStateManager.alphaFunc(516, 0.1F);
                mc.ingameGUI.renderGameOverlay(renderTickTime);
            }
        }

        mc.mcProfiler.endSection();
    }
}
