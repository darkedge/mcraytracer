package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.ScaledResolution;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.client.renderer.VertexBuffer;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.util.math.MathHelper;
import net.minecraftforge.client.event.GuiScreenEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
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
    private ScaledResolution sr;

    public static final String MODID = "mj_raytracer";
    public static final String VERSION = "1.0";

    private native void init();
    private native int resize(int width, int height);
    private native void raytrace();

    @EventHandler
    public void init(FMLInitializationEvent event) {
        mc = Minecraft.getMinecraft();
        MinecraftForge.EVENT_BUS.register(this);
        init();
    }

    @SubscribeEvent
    public void resize(GuiScreenEvent.InitGuiEvent.Pre event) {
        if (displayWidth != mc.displayWidth || displayHeight != mc.displayHeight) {
            displayWidth = mc.displayWidth;
            displayHeight = mc.displayHeight;
            //System.out.println("Resize event!");
            texture = resize(displayWidth, displayHeight);
            sr = new ScaledResolution(this.mc);
        }
    }

    @SubscribeEvent
    public void render(TickEvent.RenderTickEvent event) {
        if (event.phase == TickEvent.Phase.START) {
            // Run raytracer
            raytrace();

            GlStateManager.bindTexture(texture);

            double height = sr.getScaledHeight_double();
            double width = sr.getScaledWidth_double();
            if (width > height) {
                width *= (width / height);
            } else {
                height *= (height / width);
            }

            Tessellator tessellator = Tessellator.getInstance();
            VertexBuffer vertexbuffer = tessellator.getBuffer();
            GlStateManager.enableBlend();
            GlStateManager.tryBlendFuncSeparate(GlStateManager.SourceFactor.SRC_ALPHA, GlStateManager.DestFactor.ONE_MINUS_SRC_ALPHA, GlStateManager.SourceFactor.ONE, GlStateManager.DestFactor.ZERO);
            vertexbuffer.begin(7, DefaultVertexFormats.POSITION_TEX);
            vertexbuffer.pos(0.0, height, 0.0).tex(0.0, 0.0).endVertex();
            vertexbuffer.pos(width, height, 0.0).tex(1.0, 0.0).endVertex();
            vertexbuffer.pos(width, 0.0, 0.0).tex(1.0, 1.0).endVertex();
            vertexbuffer.pos(0.0, 0.0, 0.0).tex(0.0, 1.0).endVertex();
            tessellator.draw();
            GlStateManager.disableBlend();

            GlStateManager.pushMatrix();
            String splashText = "Hello world!";
            mc.fontRendererObj.drawString(splashText, 0, 1, -256);
            GlStateManager.popMatrix();

            // Render overlay (Inventory, Menu)
            renderGameOverlay(event.renderTickTime);

            // Save the WorldClient
            wc = mc.theWorld;
            mc.theWorld = null;
        } else if (event.phase == TickEvent.Phase.END) {
            // Restore the WorldClient
            mc.theWorld = wc;
            wc = null;
        }
    }

    // TODO: Overlays currently have dirt background instead of the game
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
