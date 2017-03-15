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

// http://www.minecraftforum.net/forums/mapping-and-modding/mapping-and-modding-tutorials/1571568-tutorial-1-6-2-changing-vanilla-without-editing
// TODO: srgnames for class methods
public class ClassTransformer implements IClassTransformer {

    private byte[] patchEntityRenderer(String name, byte[] basicClass, boolean obfuscated) {
        System.out.println("Transforming " + name + ".");

        boolean success = false;

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            if (methodNode.name.equals("updateCameraAndRender")) {
                // Insert hook for Raytracer
                InsnList hook = new InsnList();

                hook.add(new MethodInsnNode(Opcodes.INVOKESTATIC, "com/marcojonkers/mcraytracer/Raytracer", "getRaytracer", "()Lcom/marcojonkers/mcraytracer/Raytracer;", false));
                hook.add(new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "com/marcojonkers/mcraytracer/Raytracer", "onRenderTickEvent", "()V", false));

                methodNode.instructions.insert(methodNode.instructions.get(359), hook);
                success = true;
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
        System.out.println("Transforming " + name + ".");

        boolean success = false;

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<clinit>")) {

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
        System.out.println("Transforming " + name + ".");

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        // Change three fields
        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("starVBO") || fieldNode.name.equals("skyVBO") || fieldNode.name.equals("sky2VBO")) {
                //fieldNode.desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
            }
        }

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<init>")) {
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/VboRenderList", "com/marcojonkers/mcraytracer/RaytracerRenderList");
            }
            if (methodNode.name.equals("generateSky2")) {
                //changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
            if (methodNode.name.equals("generateSky")) {
                //changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
            if (methodNode.name.equals("generateStars")) {
                //changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
            if (methodNode.name.equals("renderSky")) {
                //changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
        }

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private String internalNameToDescriptor(String input, boolean array) {
        StringBuilder sb = new StringBuilder();
        if (array) sb.append('[');
        sb.append('L');
        sb.append(input);
        sb.append(';');
        return sb.toString();
    }

    // input and output are internal names
    private void changeMethodInstructions(MethodNode node, String input, String output) {
        Iterator<AbstractInsnNode> it = node.instructions.iterator();

        String inputDesc = internalNameToDescriptor(input, false);
        String outputDesc = internalNameToDescriptor(output, false);
        String inputDescArray = internalNameToDescriptor(input, true);
        String outputDescArray = internalNameToDescriptor(output, true);

        while (it.hasNext()) {
            AbstractInsnNode insn = it.next();
            switch (insn.getOpcode()) {
                // TypeInsnNode
                case Opcodes.NEW:
                case Opcodes.ANEWARRAY:
                case Opcodes.CHECKCAST:
                case Opcodes.INSTANCEOF:
                    TypeInsnNode typeInsnNode = (TypeInsnNode) insn;
                    if (typeInsnNode.desc.equals(input)) {
                        typeInsnNode.desc = output;
                    }
                    break;
                // FieldInsnNode
                case Opcodes.GETSTATIC:
                case Opcodes.PUTSTATIC:
                case Opcodes.GETFIELD:
                case Opcodes.PUTFIELD:
                    FieldInsnNode fieldInsnNode = (FieldInsnNode) insn;
                    if (fieldInsnNode.desc.equals(inputDesc)) {
                        fieldInsnNode.desc = outputDesc;
                    } else if (fieldInsnNode.desc.equals(inputDescArray)) {
                        fieldInsnNode.desc = outputDescArray;
                    }
                    break;
                // MethodInsnNode
                case Opcodes.INVOKEVIRTUAL:
                case Opcodes.INVOKESPECIAL:
                case Opcodes.INVOKESTATIC:
                case Opcodes.INVOKEINTERFACE:
                    MethodInsnNode methodInsnNode = (MethodInsnNode) insn;
                    if (methodInsnNode.owner.equals(input)) {
                        methodInsnNode.owner = output;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private byte[] patchRenderChunk(String name, byte[] basicClass, boolean obfuscated) {
        System.out.println("Transforming " + name + ".");

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("vertexBuffers")) {
                fieldNode.desc = "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                break;
            }
        }

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<init>")) {
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
            if (methodNode.name.equals("getVertexBufferByLayer")) {
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
                methodNode.desc = "(I)Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
            }
            if (methodNode.name.equals("deleteGlResources")) {
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer");
            }
        }

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchChunkRenderDispatcher(String name, byte[] basicClass, boolean obfuscated) {
        System.out.println("Transforming " + name + ".");

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("uploadChunk")) {
                // INVOKEVIRTUAL net/minecraft/client/renderer/chunk/RenderChunk.getVertexBufferByLayer (I)Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((MethodInsnNode) methodNode.instructions.get(16)).desc = "(I)Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // Load RenderChunk local variable
                methodNode.instructions.insert(methodNode.instructions.get(16), new VarInsnNode(Opcodes.ALOAD, 3));
                // INVOKESPECIAL net/minecraft/client/renderer/chunk/ChunkRenderDispatcher.uploadVertexBuffer (Lnet/minecraft/client/renderer/VertexBuffer;Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(17 + 1)).desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;Lnet/minecraft/client/renderer/chunk/RenderChunk;)V";
            }
            if (methodNode.name.equals("uploadVertexBuffer")) {
                methodNode.desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;Lnet/minecraft/client/renderer/chunk/RenderChunk;)V";
                //methodNode.maxLocals++; // Unnecessary due to ClassWriter.COMPUTE_MAXS?
                methodNode.localVariables.add(new LocalVariableNode(
                                "foo",
                                "Lnet/minecraft/client/renderer/chunk/RenderChunk;",
                                null,
                                (LabelNode) methodNode.instructions.get(0), // L0
                                (LabelNode) methodNode.instructions.get(15), // L3
                                3
                        )
                );
                // INVOKEVIRTUAL net/minecraft/client/renderer/VertexBufferUploader.setVertexBuffer (Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(5)).desc = "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";

                // Add RenderChunk param to draw call
                methodNode.instructions.insert(methodNode.instructions.get(10), new VarInsnNode(Opcodes.ALOAD, 3));
                // INVOKEVIRTUAL net/minecraft/client/renderer/VertexBufferUploader.draw (Lnet/minecraft/client/renderer/VertexBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(11 + 1)).desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lnet/minecraft/client/renderer/chunk/RenderChunk;)V";
            }
        }

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchVertexBufferUploader(String name, byte[] basicClass, boolean obfuscated) {
        System.out.println("Transforming " + name + ".");

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("vertexBuffer")) {
                fieldNode.desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                break;
            }
        }

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("draw")) {
                methodNode.desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lnet/minecraft/client/renderer/chunk/RenderChunk;)V";
                methodNode.localVariables.add(new LocalVariableNode(
                                "foo",
                                "Lnet/minecraft/client/renderer/chunk/RenderChunk;",
                                null,
                                (LabelNode) methodNode.instructions.get(0), // L0
                                (LabelNode) methodNode.instructions.get(14), // L3
                                2
                        )
                );
                // GETFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) methodNode.instructions.get(7)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";

                // Load RenderChunk local variable
                methodNode.instructions.insert(methodNode.instructions.get(9), new VarInsnNode(Opcodes.ALOAD, 2));
                // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.bufferData (Ljava/nio/ByteBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(10 + 1)).owner = "com/marcojonkers/mcraytracer/CppVertexBuffer";
                ((MethodInsnNode) methodNode.instructions.get(10 + 1)).desc = "(Ljava/nio/ByteBuffer;Lnet/minecraft/client/renderer/chunk/RenderChunk;)V";
            }
            if (methodNode.name.equals("setVertexBuffer")) {
                methodNode.desc = "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";
                // PUTFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) methodNode.instructions.get(4)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
            }
        }

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    @Override
    // TODO: Obfuscated names
    public byte[] transform(String name, String transformedName, byte[] basicClass) {
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

        if (name.equals("net.minecraft.client.renderer.chunk.RenderChunk")) {
            return patchRenderChunk(name, basicClass, false);
        }

        if (name.equals("net.minecraft.client.renderer.chunk.ChunkRenderDispatcher")) {
            return patchChunkRenderDispatcher(name, basicClass, false);
        }

        if (name.equals("net.minecraft.client.renderer.VertexBufferUploader")) {
            return patchVertexBufferUploader(name, basicClass, false);
        }

        return basicClass;
    }
}
