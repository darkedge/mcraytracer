package com.marcojonkers.mcraytracer;

import net.minecraft.client.renderer.ChunkRenderContainer;
import net.minecraft.client.renderer.chunk.RenderChunk;
import net.minecraft.client.renderer.vertex.VertexBuffer;
import net.minecraft.util.BlockRenderLayer;
import net.minecraft.util.math.BlockPos;

public class RaytracerRenderList extends ChunkRenderContainer {
    public RaytracerRenderList() {
        System.out.println("RaytracerRenderList WORKS!");
    }

    @Override
    public void renderChunkLayer(BlockRenderLayer layer) {
        if (this.initialized) {
            for (RenderChunk renderchunk : this.renderChunks)
            {
                VertexBuffer buffer = renderchunk.getVertexBufferByLayer(layer.ordinal());
                BlockPos pos = renderchunk.getPosition();
                Raytracer.getRaytracer().setVertexBuffer(pos.getX(), pos.getY(), pos.getZ(), buffer);
            }
        }
    }
}
