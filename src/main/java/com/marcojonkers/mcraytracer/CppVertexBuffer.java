package com.marcojonkers.mcraytracer;

import net.minecraft.client.renderer.chunk.RenderChunk;
import net.minecraft.client.renderer.vertex.VertexFormat;
import net.minecraft.util.BlockRenderLayer;
import net.minecraft.util.math.BlockPos;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

import java.nio.ByteBuffer;

@SideOnly(Side.CLIENT)
public class CppVertexBuffer {
    private VertexFormat format; // Probably constant
    private BlockPos blockPos;
    private BlockRenderLayer layer;

    public CppVertexBuffer(VertexFormat vertexFormatIn) {
        this.format = vertexFormatIn;
    }

    public void bindBuffer() {
        // Do nothing
    }

    public void bufferData(ByteBuffer data, RenderChunk renderChunk, BlockRenderLayer layer) {
        // Pass to C++
        this.blockPos = renderChunk.getPosition();
        this.layer = layer;
        Raytracer.getRaytracer().setVertexBuffer(blockPos.getX(), blockPos.getY(), blockPos.getZ(), layer.ordinal(), data, data.remaining());
    }

    public void drawArrays(int mode) {
        // Do nothing
    }

    public void unbindBuffer() {
        // Do nothing
    }

    public void deleteGlBuffers() {
        Raytracer.getRaytracer().deleteVertexBuffer(blockPos.getX(), blockPos.getY(), blockPos.getZ(), layer.ordinal());
    }
}
