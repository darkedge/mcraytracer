package com.marcojonkers.mcraytracer;

import net.minecraft.client.renderer.vertex.VertexFormat;
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

    public void bufferData(ByteBuffer data) {
        // Pass to C++
        Raytracer.getRaytracer().setVertexBuffer(data, data.remaining());
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
