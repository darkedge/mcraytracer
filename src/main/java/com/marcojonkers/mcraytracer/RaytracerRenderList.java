package com.marcojonkers.mcraytracer;

import com.google.common.collect.Sets;
import net.minecraft.client.renderer.ChunkRenderContainer;
import net.minecraft.client.renderer.chunk.RenderChunk;
import net.minecraft.client.renderer.vertex.VertexBuffer;
import net.minecraft.util.BlockRenderLayer;
import net.minecraft.util.math.BlockPos;

import java.util.Set;

public class RaytracerRenderList extends ChunkRenderContainer {
    private Set<RenderChunk> set;
    @Override
    public void initialize(double viewEntityXIn, double viewEntityYIn, double viewEntityZIn) {
        super.initialize(viewEntityXIn, viewEntityYIn, viewEntityZIn);
        if (set == null) {
            set = Sets.newHashSetWithExpectedSize(17424);
        }
        set.clear();
        Raytracer.getRaytracer().setViewEntity(viewEntityXIn, viewEntityYIn, viewEntityZIn);
    }

    @Override
    public void renderChunkLayer(BlockRenderLayer layer) {
        if (this.initialized) {
            int i = layer.ordinal();
            if (i != 0) return;
            for (RenderChunk renderchunk : this.set) {
                VertexBuffer buffer = renderchunk.getVertexBufferByLayer(i);
                BlockPos pos = renderchunk.getPosition();
                Raytracer.getRaytracer().setVertexBuffer(pos, i, buffer);
            }
        }
    }

    @Override
    public void addRenderChunk(RenderChunk renderChunkIn, BlockRenderLayer layer) {
        this.set.add(renderChunkIn);
        //super.addRenderChunk(renderChunkIn, layer);
    }
}
