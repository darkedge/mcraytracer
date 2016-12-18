package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.gui.GuiVideoSettings;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.*;
import net.minecraft.client.renderer.chunk.ListChunkFactory;
import net.minecraft.client.renderer.chunk.VboChunkFactory;
import net.minecraft.client.renderer.vertex.VertexBuffer;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.world.chunk.Chunk;
import net.minecraftforge.client.event.GuiOpenEvent;
import net.minecraftforge.client.event.GuiScreenEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;
import org.lwjgl.util.vector.Vector3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

@Mod(modid = Raytracer.MODID, version = Raytracer.VERSION)
public class Raytracer {
    private static Raytracer raytracer;

    static {
        System.loadLibrary("loader");
    }

    private Renderer renderer;

    private Minecraft mc;
    private WorldClient wc;

    private int displayWidth;
    private int displayHeight;

    private float textureWidth;
    private float textureHeight;
    private boolean enabled = true;

    public static final String MODID = "mj_raytracer";
    private static final Logger LOGGER = LogManager.getLogger(MODID);
    public static final String VERSION = "1.0";
    private static final KeyBinding TOGGLE_KEY = new KeyBinding("Toggle Ray Tracing", Keyboard.KEY_G, MODID);

    // C++ functions
    private native void init();
    private native void resize(int width, int height);
    private native int raytrace();
    public native void setViewingPlane(FloatBuffer buffer);
    private native void setVertexBuffer(int x, int y, int z, VertexBuffer buffer);
    public native void setViewEntity(double x, double y, double z);

    // TODO: Group calls together to prevent JNI overhead
    public void setVertexBuffer(BlockPos pos, VertexBuffer buffer) {
        if (enabled) {
            setVertexBuffer(pos.getX(), pos.getY(), pos.getZ(), buffer);
        }
    }

    public ChunkRenderContainer renderContainer;

    public Raytracer() {
        raytracer = this;
        mc = Minecraft.getMinecraft();
        MinecraftForge.EVENT_BUS.register(this);
        ClientRegistry.registerKeyBinding(TOGGLE_KEY);
        renderer = new Renderer(this);

        this.renderContainer = new RaytracerRenderList();

        init();
    }

    public static Raytracer getRaytracer() {
        return raytracer;
    }

    @SubscribeEvent
    public void onClientTickEvent(TickEvent.ClientTickEvent event) {
        if (TOGGLE_KEY.isPressed()) {
            enabled = !enabled;
        }
    }

    /**
     * Resize the OpenGL/CUDA resources
     * @param event
     */
    @SubscribeEvent
    public void onPreInitGuiEvent(GuiScreenEvent.InitGuiEvent.Pre event) {
        if (displayWidth != mc.displayWidth || displayHeight != mc.displayHeight) {
            LOGGER.info("Java: Resize");

            displayWidth = mc.displayWidth;
            displayHeight = mc.displayHeight;
            resize(displayWidth, displayHeight);

            textureWidth = (float)Math.pow(2.0, Math.ceil(Math.log((double) displayWidth) / Math.log(2.0)));
            textureHeight = (float)Math.pow(2.0, Math.ceil(Math.log((double) displayHeight) / Math.log(2.0)));
        }
    }

    public Vector3f getViewVector(Entity entityIn, double partialTicks) {
        float f = (float)((double)entityIn.prevRotationPitch + (double)(entityIn.rotationPitch - entityIn.prevRotationPitch) * partialTicks);
        float f1 = (float)((double)entityIn.prevRotationYaw + (double)(entityIn.rotationYaw - entityIn.prevRotationYaw) * partialTicks);

        if (Minecraft.getMinecraft().gameSettings.thirdPersonView == 2) {
            f += 180.0F;
        }

        float f2 = MathHelper.cos(-f1 * 0.017453292F - (float)Math.PI);
        float f3 = MathHelper.sin(-f1 * 0.017453292F - (float)Math.PI);
        float f4 = -MathHelper.cos(-f * 0.017453292F);
        float f5 = MathHelper.sin(-f * 0.017453292F);
        return new Vector3f(f3 * f4, f5, f2 * f4);
    }

    public void onRenderTickEvent() {
        if (!enabled) return;
        Entity entity = this.mc.getRenderViewEntity();
        if (entity == null) return;

        renderer.updateCameraAndRender();

        // Run raytracer
        int texture = raytrace();

        // Draw result to screen
        GlStateManager.bindTexture(texture);

        GlStateManager.enableBlend();
        GL11.glMatrixMode(GL11.GL_MODELVIEW);
        GL11.glPushMatrix();
        GL11.glLoadIdentity();
        GL11.glMatrixMode(GL11.GL_PROJECTION);
        GL11.glPushMatrix();
        GL11.glLoadIdentity();
        GL11.glOrtho(0.0, (double) this.mc.displayWidth, (double) this.mc.displayHeight, 0.0, -1.0, 1.0);
        GL11.glBegin(GL11.GL_QUADS);

        GL11.glTexCoord2f(0.0f, 0.0f); GL11.glVertex2f(0.0f, textureHeight);
        GL11.glTexCoord2f(1.0f, 0.0f); GL11.glVertex2f(textureWidth, textureHeight);
        GL11.glTexCoord2f(1.0f, 1.0f); GL11.glVertex2f(textureWidth, 0.0f);
        GL11.glTexCoord2f(0.0f, 1.0f); GL11.glVertex2f(0.0f, 0.0f);

        GL11.glEnd();
        GL11.glPopMatrix();
        GL11.glMatrixMode(GL11.GL_MODELVIEW);
        GL11.glPopMatrix();
    }
}
