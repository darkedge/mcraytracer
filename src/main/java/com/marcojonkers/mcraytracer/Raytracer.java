package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.gui.GuiVideoSettings;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.Matrix4f;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
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
    static {
        System.loadLibrary("loader");
    }

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
    private native void loadChunk(Chunk chunk);
    private native void setViewingPlane(FloatBuffer buffer);

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

    /**
     * Needed to prevent GUI screens showing the default dirt background
     * @param event
     */
    @SubscribeEvent
    public void onPreDrawScreenEvent(GuiScreenEvent.DrawScreenEvent.Pre event) {
        if (enabled) {
            restoreTheWorld();
        }
    }

    /**
     * Needed to prevent GUI screens showing the default dirt background
     * @param event
     */
    @SubscribeEvent
    public void onPostDrawScreenEvent(GuiScreenEvent.DrawScreenEvent.Post event) {
        if (enabled) {
            takeOverTheWorld();
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

    private Matrix4f glhFrustumf2(float left, float right, float bottom, float top, float znear, float zfar) {
        float temp = 2.0f * znear;
        float temp2 = right - left;
        float temp3 = top - bottom;
        float temp4 = zfar - znear;

        return new Matrix4f(new float[]{
                temp / temp2,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                temp / temp3,
                0.0f,
                0.0f,
                (right + left) / temp2,
                (top + bottom) / temp3,
                (-zfar - znear) / temp4,
                -1.0f,
                0.0f,
                0.0f,
                (-temp * zfar) / temp4,
                0.0f
        });
    }

    private Matrix4f glhPerspectivef2(float fovy, float aspect, float znear, float zfar) {
        float ymax, xmax;
        ymax = znear * (float) Math.tan(fovy * Math.PI / 360.0);
        xmax = ymax * aspect;
        return glhFrustumf2(-xmax, xmax, -ymax, ymax, znear, zfar);
    }

    private void updateCameraAndRender() {
        // Set camera
        Entity entity = this.mc.getRenderViewEntity();
        if (entity == null) return;
        float partialTicks = this.mc.getRenderPartialTicks();

        float fov = this.mc.gameSettings.fovSetting;

        // Build projection matrix

        // TODO: Camera zoom
        // see: EntityRenderer.setupCameraTransform()

        float farPlaneDistance = (float)(this.mc.gameSettings.renderDistanceChunks * 16);
        Matrix4f projection = glhPerspectivef2(fov, (float)this.mc.displayWidth / (float)this.mc.displayHeight, 0.05F, farPlaneDistance * MathHelper.SQRT_2);

        // Build view matrix

        // TODO: Lots of stuff (bobbing, portal, hurting, sleeping, 3rd person, etc etc)
        // see: EntityRenderer.orientCamera()

        Matrix4f view = new Matrix4f();
        view.setIdentity();
        view.translate(new Vector3f(0.0f, 0.0f, 0.05f));

        float yaw = entity.prevRotationYaw + (entity.rotationYaw - entity.prevRotationYaw) * partialTicks + 180.0F;
        float pitch = entity.prevRotationPitch + (entity.rotationPitch - entity.prevRotationPitch) * partialTicks;

        view.rotate(pitch, new Vector3f(1.0f, 0.0f, 0.0f));
        view.rotate(yaw, new Vector3f(0.0f, 1.0f, 0.0f));
        view.translate(new Vector3f(0.0f, -entity.getEyeHeight(), 0.0f));

        FloatBuffer modelMatrix = FloatBuffer.allocate(16);
        view.store(modelMatrix); modelMatrix.position(0);
        FloatBuffer projMatrix = FloatBuffer.allocate(16);
        projection.store(projMatrix); projMatrix.position(0);
        IntBuffer viewport = IntBuffer.allocate(4);
        viewport.put(new int[]{0, 0, this.displayWidth, this.displayHeight}); viewport.position(0);
        FloatBuffer obj_pos = ByteBuffer.allocateDirect(10 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        // gluUnProject does not advance any buffer positions
        Project.gluUnProject(0.0f, 0.0f, 0.0f, modelMatrix, projMatrix, viewport, obj_pos); obj_pos.position(obj_pos.position() + 3);
        Project.gluUnProject(this.displayWidth, 0.0f, 0.0f, modelMatrix, projMatrix, viewport, obj_pos); obj_pos.position(obj_pos.position() + 3);
        Project.gluUnProject(0.0f, displayHeight, 0.0f, modelMatrix, projMatrix, viewport, obj_pos); obj_pos.position(obj_pos.position() + 3);
        obj_pos.put(fov);

        setViewingPlane(obj_pos);
    }

    /**
     * Replace default video settings screen with custom raytracer options menu
     * @param event
     */
    @SubscribeEvent
    public void onGuiOpenEvent(GuiOpenEvent event) {
        GuiScreen gui = event.getGui();
        if (gui instanceof GuiVideoSettings) {
            event.setGui(new GuiRaytracerSettings(mc.currentScreen, this.mc.gameSettings));
        }
    }

    @SubscribeEvent
    public void onRenderTickEvent(TickEvent.RenderTickEvent event) {
        if (!enabled) return;
        if (event.phase == TickEvent.Phase.START) {
            this.updateCameraAndRender();

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
        mc.mcProfiler.startSection("gui");

        if (mc.thePlayer != null) {
            if (!mc.gameSettings.hideGUI || mc.currentScreen != null) {
                GlStateManager.alphaFunc(516, 0.1F);
                mc.ingameGUI.renderGameOverlay(renderTickTime);
            }
        }

        mc.mcProfiler.endSection();
    }
}
