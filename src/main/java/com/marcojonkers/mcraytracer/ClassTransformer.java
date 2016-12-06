package com.marcojonkers.mcraytracer;

import jdk.internal.org.objectweb.asm.ClassReader;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.tree.AbstractInsnNode;
import jdk.internal.org.objectweb.asm.tree.ClassNode;
import jdk.internal.org.objectweb.asm.tree.MethodNode;
import net.minecraft.launchwrapper.IClassTransformer;

import java.util.Iterator;

import static jdk.internal.org.objectweb.asm.Opcodes.FDIV;

public class ClassTransformer implements IClassTransformer {
    private boolean obfuscated;

    // http://www.minecraftforum.net/forums/mapping-and-modding/mapping-and-modding-tutorials/1571568-tutorial-1-6-2-changing-vanilla-without-editing
    private byte[] patch(String name, String transformedName, byte[] basicClass) {
        return basicClass;
    }

    @Override
    public byte[] transform(String name, String transformedName, byte[] basicClass) {
        // Obfuscated
        if (name.equals("boo")) {
            System.out.println("Found obfuscated version of VboRenderList.");
            return patch(name, transformedName, basicClass);
        }

        // Non-obfuscated
        if (name.equals("net.minecraft.client.renderer.VboRenderList")) {
            System.out.println("Found non-obfuscated version of VboRenderList.");
            return patch(name, transformedName, basicClass);
        }

        return basicClass;
    }
}
