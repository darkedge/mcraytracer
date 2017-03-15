package com.marcojonkers.mcraytracer;

import net.minecraft.client.renderer.block.model.BlockPart;
import net.minecraft.client.renderer.chunk.RenderChunk;
import net.minecraft.client.renderer.vertex.VertexFormat;
import net.minecraft.util.math.BlockPos;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

import java.nio.ByteBuffer;

@SideOnly(Side.CLIENT)
public class CppVertexBuffer {
    VertexFormat format;

    public CppVertexBuffer(VertexFormat vertexFormatIn) {
        this.format = vertexFormatIn;
    }

    public void bindBuffer() {
        // Do nothing
    }

    public void bufferData(ByteBuffer data, RenderChunk renderChunk) {
        // Pass to C++
        BlockPos blockPos = renderChunk.getPosition();
        Raytracer.getRaytracer().setVertexBuffer(blockPos.getX(), blockPos.getY(), blockPos.getZ(), data, data.remaining());
    }

    public void drawArrays(int mode) {
        // Do nothing
    }

    public void unbindBuffer() {
        // Do nothing
    }

    public void deleteGlBuffers() {

    }
}
