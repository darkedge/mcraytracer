package com.marcojonkers.mcraytracer;

import jdk.internal.org.objectweb.asm.ClassReader;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.tree.AbstractInsnNode;
import jdk.internal.org.objectweb.asm.tree.ClassNode;
import jdk.internal.org.objectweb.asm.tree.MethodInsnNode;
import jdk.internal.org.objectweb.asm.tree.MethodNode;
import net.minecraft.launchwrapper.IClassTransformer;

import java.util.Iterator;

public class ClassTransformer implements IClassTransformer {
    // http://www.minecraftforum.net/forums/mapping-and-modding/mapping-and-modding-tutorials/1571568-tutorial-1-6-2-changing-vanilla-without-editing
    private byte[] patch(String name, String transformedName, byte[] basicClass, boolean obfuscated) {
        String targetMethodName = "";
        if (obfuscated) {
            targetMethodName = "a";
        } else {
            targetMethodName = "renderChunkLayer";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        Iterator<MethodNode> methods = classNode.methods.iterator();
        while (methods.hasNext()) {
            MethodNode methodNode = methods.next();
            // Compare method
            if (methodNode.name.equals(targetMethodName)) {
                System.out.println("Found renderChunkLayer().");

                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();

                int insnIndex = 0;
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction instanceof MethodInsnNode) {
                        MethodInsnNode methodinsn = (MethodInsnNode) instruction;
                        if (methodinsn.name.equals("drawArrays")) {
                            System.out.println("Found drawArrays().");
                            break;
                        }
                    }
                    insnIndex++;
                }

                AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex-2); // ALOAD 4
                AbstractInsnNode remNode2 = methodNode.instructions.get(insnIndex-1); // BIPUSH 7
                AbstractInsnNode remNode3 = methodNode.instructions.get(insnIndex); // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.drawArrays (I)V

                methodNode.instructions.remove(remNode1);
                methodNode.instructions.remove(remNode2);
                methodNode.instructions.remove(remNode3);

                // Stop looking for methods
                break;
            }
        }

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    @Override
    public byte[] transform(String name, String transformedName, byte[] basicClass) {
        // Obfuscated
        if (name.equals("boo")) {
            System.out.println("Found obfuscated version of VboRenderList.");
            return patch(name, transformedName, basicClass, true);
        }

        // Non-obfuscated
        if (name.equals("net.minecraft.client.renderer.VboRenderList")) {
            System.out.println("Found non-obfuscated version of VboRenderList.");
            return patch(name, transformedName, basicClass, false);
        }

        return basicClass;
    }
}
