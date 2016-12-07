package com.marcojonkers.mcraytracer;

import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.Matrix4f;
import net.minecraft.client.renderer.ViewFrustum;
import net.minecraft.client.renderer.chunk.IRenderChunkFactory;
import net.minecraft.client.renderer.chunk.VboChunkFactory;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.MathHelper;
import org.lwjgl.util.glu.Project;
import org.lwjgl.util.vector.Vector3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class Renderer {
    private Raytracer raytracer;
    private Minecraft mc;
    private WorldClient wc;

    private int renderDistanceChunks;
    private ViewFrustum viewFrustum;
    private IRenderChunkFactory renderChunkFactory;

    public Renderer(Raytracer raytracer) {
        this.raytracer = raytracer;
        mc = Minecraft.getMinecraft();
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

    private void setupTerrain() {
        if (this.mc.gameSettings.renderDistanceChunks != this.renderDistanceChunks) {
            if (this.viewFrustum != null) {
                this.viewFrustum.deleteGlResources();
            }
            this.renderDistanceChunks = this.mc.gameSettings.renderDistanceChunks;
            this.renderChunkFactory = new VboChunkFactory();
            this.viewFrustum = new ViewFrustum(this.wc, this.mc.gameSettings.renderDistanceChunks, this.mc.renderGlobal, this.renderChunkFactory);
        }
    }

    public void updateCameraAndRender() {
        // Set camera
        Entity entity = this.mc.getRenderViewEntity();
        if (entity == null) return;
        float partialTicks = this.mc.getRenderPartialTicks();

        float fov = this.mc.gameSettings.fovSetting;

        // Build projection matrix

        // TODO: Camera zoom
        // see: EntityRenderer.setupCameraTransform()

        float farPlaneDistance = (float) (this.mc.gameSettings.renderDistanceChunks * 16);
        Matrix4f projection = glhPerspectivef2(fov, (float) this.mc.displayWidth / (float) this.mc.displayHeight, 0.05F, farPlaneDistance * MathHelper.SQRT_2);

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
        view.store(modelMatrix);
        modelMatrix.position(0);
        FloatBuffer projMatrix = FloatBuffer.allocate(16);
        projection.store(projMatrix);
        projMatrix.position(0);
        IntBuffer viewport = IntBuffer.allocate(4);
        viewport.put(new int[]{0, 0, this.mc.displayWidth, this.mc.displayHeight});
        viewport.position(0);
        FloatBuffer obj_pos = ByteBuffer.allocateDirect(10 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        // gluUnProject does not advance any buffer positions
        Project.gluUnProject(0.0f, 0.0f, 0.0f, modelMatrix, projMatrix, viewport, obj_pos);
        obj_pos.position(obj_pos.position() + 3);
        Project.gluUnProject(this.mc.displayWidth, 0.0f, 0.0f, modelMatrix, projMatrix, viewport, obj_pos);
        obj_pos.position(obj_pos.position() + 3);
        Project.gluUnProject(0.0f, this.mc.displayHeight, 0.0f, modelMatrix, projMatrix, viewport, obj_pos);
        obj_pos.position(obj_pos.position() + 3);
        obj_pos.put(fov);

        raytracer.setViewingPlane(obj_pos);

        setupTerrain();
    }
}
