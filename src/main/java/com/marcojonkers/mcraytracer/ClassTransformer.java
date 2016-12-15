package com.marcojonkers.mcraytracer;

import jdk.internal.org.objectweb.asm.ClassReader;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.Opcodes;
import jdk.internal.org.objectweb.asm.Type;
import jdk.internal.org.objectweb.asm.tree.*;
import jdk.internal.org.objectweb.asm.util.Printer;
import jdk.internal.org.objectweb.asm.util.Textifier;
import jdk.internal.org.objectweb.asm.util.TraceMethodVisitor;
import net.minecraft.client.settings.GameSettings;
import net.minecraft.launchwrapper.IClassTransformer;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Iterator;

public class ClassTransformer implements IClassTransformer {

    // http://www.minecraftforum.net/forums/mapping-and-modding/mapping-and-modding-tutorials/1571568-tutorial-1-6-2-changing-vanilla-without-editing
    private byte[] patchVboRenderList(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = false;
        String targetMethodName;
        if (obfuscated) {
            targetMethodName = "a";
        } else {
            targetMethodName = "renderChunkLayer";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
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

                success = true;

                // Stop looking for methods
                break;
            }
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchEntityRenderer(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = false;
        String targetMethodName;
        if (obfuscated) {
            targetMethodName = "TODO";
        } else {
            targetMethodName = "updateCameraAndRender";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals(targetMethodName)) {
                System.out.println("Found updateCameraAndRender().");

                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction instanceof MethodInsnNode) {
                        MethodInsnNode methodinsn = (MethodInsnNode) instruction;
                        if (methodinsn.name.equals("renderWorld")) {
                            targetNode = instruction;
                            System.out.println("Found renderWorld().");
                            break;
                        }
                    }
                }

                // Insert hook for Raytracer
                InsnList hook = new InsnList();

                hook.add(new MethodInsnNode(Opcodes.INVOKESTATIC, "com/marcojonkers/mcraytracer/Raytracer", "getRaytracer", "()Lcom/marcojonkers/mcraytracer/Raytracer;", false));
                hook.add(new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "com/marcojonkers/mcraytracer/Raytracer", "onRenderTickEvent", "()V", false));

                methodNode.instructions.insert(targetNode, hook);

                success = true;

                // Stop looking for methods
                break;
            }
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private static Printer printer = new Textifier();
    private static TraceMethodVisitor mp = new TraceMethodVisitor(printer);
    public static String insnToString(AbstractInsnNode insn){
        insn.accept(mp);
        StringWriter sw = new StringWriter();
        printer.print(new PrintWriter(sw));
        printer.getText().clear();
        return sw.toString();
    }

    private byte[] patchGuiVideoSettings(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = false;

        String targetMethodName;
        if (obfuscated) {
            targetMethodName = "TODO";
        } else {
            targetMethodName = "VIDEO_OPTIONS";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<clinit>")) {
                System.out.println("Found <clinit>.");

                methodNode.instructions.clear();

                GameSettings.Options[] options = new GameSettings.Options[]{
                        GameSettings.Options.RENDER_DISTANCE,
                        GameSettings.Options.FRAMERATE_LIMIT,
                        GameSettings.Options.VIEW_BOBBING,
                        GameSettings.Options.GUI_SCALE,
                        GameSettings.Options.ATTACK_INDICATOR,
                        GameSettings.Options.USE_FULLSCREEN,
                        GameSettings.Options.ENABLE_VSYNC,
                        GameSettings.Options.MIPMAP_LEVELS
                };
                InsnList list = new InsnList();
                list.add(new IntInsnNode(Opcodes.BIPUSH, 8));
                list.add(new TypeInsnNode(Opcodes.ANEWARRAY, Type.getInternalName(GameSettings.Options.class)));
                for (int i = 0; i < options.length; i++) {
                    list.add(new InsnNode(Opcodes.DUP));
                    list.add(new IntInsnNode(Opcodes.BIPUSH, i));
                    list.add(new FieldInsnNode(Opcodes.GETSTATIC, Type.getInternalName(GameSettings.Options.class), options[i].name(), Type.getDescriptor(GameSettings.Options.class)));
                    list.add(new InsnNode(Opcodes.AASTORE));
                }
                // This throws java.lang.ClassCircularityError
                //list.add(new FieldInsnNode(Opcodes.PUTSTATIC, Type.getInternalName(GuiVideoSettings.class), targetMethodName, Type.getDescriptor(options.getClass())));
                // So hard-code it for now
                list.add(new FieldInsnNode(Opcodes.PUTSTATIC, "net/minecraft/client/gui/GuiVideoSettings", "VIDEO_OPTIONS", "[Lnet/minecraft/client/settings/GameSettings$Options;"));
                list.add(new InsnNode(Opcodes.RETURN));

                methodNode.instructions.insert(list);

                success = true;

                // Stop looking for methods
                break;
            }
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchRenderGlobal(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = false;

        String targetMethodName;
        if (obfuscated) {
            targetMethodName = "TODO";
        } else {
            targetMethodName = "VboRenderList";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<init>")) {
                System.out.println("Found <init>.");

                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.NEW) {
                        TypeInsnNode node = (TypeInsnNode) instruction;
                        if (node.desc.equals("net/minecraft/client/renderer/VboRenderList")) {
                            System.out.println("Found VboRenderList.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // NEW net/minecraft/client/renderer/VboRenderList
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 2); // INVOKESPECIAL net/minecraft/client/renderer/VboRenderList.<init> ()V

                    methodNode.instructions.insert(remNode0, new TypeInsnNode(Opcodes.NEW, "com/marcojonkers/mcraytracer/RaytracerRenderList"));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new MethodInsnNode(Opcodes.INVOKESPECIAL, "com/marcojonkers/mcraytracer/RaytracerRenderList", "<init>", "()V", false));
                    methodNode.instructions.remove(remNode1);

                    success = true;
                }

                // Stop looking for methods
                break;
            }
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    @Override
    // TODO: Obfuscated names
    public byte[] transform(String name, String transformedName, byte[] basicClass) {
        if (name.equals("boo")) {
            return patchVboRenderList(name, basicClass, true);
        }
        if (name.equals("net.minecraft.client.renderer.VboRenderList")) {
            return patchVboRenderList(name, basicClass, false);
        }

        if (name.equals("bnz")) {
            return patchEntityRenderer(name, basicClass, true);
        }
        if (name.equals("net.minecraft.client.renderer.EntityRenderer")) {
            return patchEntityRenderer(name, basicClass, false);
        }

        if (name.equals("bgb")) {
            return patchGuiVideoSettings(name, basicClass, true);
        }
        if (name.equals("net.minecraft.client.gui.GuiVideoSettings")) {
            return patchGuiVideoSettings(name, basicClass, false);
        }

        if (name.equals("boh")) {
            return patchRenderGlobal(name, basicClass, true);
        }
        if (name.equals("net.minecraft.client.renderer.RenderGlobal")) {
            return patchRenderGlobal(name, basicClass, false);
        }

        return basicClass;
    }
}
