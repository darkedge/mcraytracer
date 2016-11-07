package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.client.renderer.VertexBuffer;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;

@Mod(modid = Raytracer.MODID, version = Raytracer.VERSION)
public class Raytracer {
    static {
        System.loadLibrary("raytracer_native");
    }

    private Minecraft mc;
    private WorldClient wc;

    public static final String MODID = "mj_raytracer";
    public static final String VERSION = "1.0";

    private native void renderJNI();

    @EventHandler
    public void init(FMLInitializationEvent event) {
        this.mc = Minecraft.getMinecraft();
        MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void render(TickEvent.RenderTickEvent event) {
        if (event.phase == TickEvent.Phase.START) {
            // TODO: Remove this when raytracer works
            GlStateManager.clear(16640);

            // Obtain texture
            renderJNI();

            GlStateManager.bindTexture(0);

            // Test quad using GL11
            double height = (double) Minecraft.getMinecraft().displayHeight;
            double width = (double) Minecraft.getMinecraft().displayWidth;

            Tessellator tessellator = Tessellator.getInstance();
            VertexBuffer vertexbuffer = tessellator.getBuffer();
            GlStateManager.enableBlend();
            GlStateManager.disableTexture2D();
            GlStateManager.tryBlendFuncSeparate(GlStateManager.SourceFactor.SRC_ALPHA, GlStateManager.DestFactor.ONE_MINUS_SRC_ALPHA, GlStateManager.SourceFactor.ONE, GlStateManager.DestFactor.ZERO);
            GlStateManager.color(1.0f, 0.0f, 0.0f, 1.0f);
            vertexbuffer.begin(7, DefaultVertexFormats.POSITION);
            vertexbuffer.pos(0.0, height, 0.0D).endVertex();
            vertexbuffer.pos(width, height, 0.0D).endVertex();
            vertexbuffer.pos(width, 0.0, 0.0D).endVertex();
            vertexbuffer.pos(0.0, 0.0, 0.0D).endVertex();
            tessellator.draw();
            GlStateManager.enableTexture2D();
            GlStateManager.disableBlend();

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
        this.mc.mcProfiler.endStartSection("gui");

        if (this.mc.thePlayer != null) {
            if (!this.mc.gameSettings.hideGUI || this.mc.currentScreen != null)
            {
                GlStateManager.alphaFunc(516, 0.1F);
                this.mc.ingameGUI.renderGameOverlay(renderTickTime);
            }
        }

        this.mc.mcProfiler.endSection();
    }
}
