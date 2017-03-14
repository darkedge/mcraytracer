package com.marcojonkers.mcraytracer;

import jdk.internal.org.objectweb.asm.ClassReader;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.Opcodes;
import jdk.internal.org.objectweb.asm.Type;
import jdk.internal.org.objectweb.asm.commons.Method;
import jdk.internal.org.objectweb.asm.tree.*;
import jdk.internal.org.objectweb.asm.util.Printer;
import jdk.internal.org.objectweb.asm.util.Textifier;
import jdk.internal.org.objectweb.asm.util.TraceMethodVisitor;
import net.minecraft.client.renderer.vertex.VertexBuffer;
import net.minecraft.client.settings.GameSettings;
import net.minecraft.launchwrapper.IClassTransformer;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Iterator;

public class ClassTransformer implements IClassTransformer {

    // http://www.minecraftforum.net/forums/mapping-and-modding/mapping-and-modding-tutorials/1571568-tutorial-1-6-2-changing-vanilla-without-editing
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
        boolean success = true;

        String targetMethodName;
        if (obfuscated) {
            targetMethodName = "TODO";
        } else {
            targetMethodName = "VboRenderList";
        }

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        // Change three fields
        FieldNode starVBO = null;
        FieldNode skyVBO = null;
        FieldNode sky2VBO = null;
        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("starVBO")) {
                starVBO = fieldNode;
            } else if (fieldNode.name.equals("skyVBO")) {
                skyVBO = fieldNode;
            } else if (fieldNode.name.equals("sky2VBO")) {
                sky2VBO = fieldNode;
            }
        }

        classNode.fields.remove(starVBO);
        classNode.fields.remove(skyVBO);
        classNode.fields.remove(sky2VBO);

        classNode.fields.add(new FieldNode(
                Opcodes.ACC_PRIVATE,
                "starVBO",
                "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                null,
                null
        ));
        classNode.fields.add(new FieldNode(
                Opcodes.ACC_PRIVATE,
                "skyVBO",
                "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                null,
                null
        ));
        classNode.fields.add(new FieldNode(
                Opcodes.ACC_PRIVATE,
                "sky2VBO",
                "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                null,
                null
        ));

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
            }

            if (methodNode.name.equals("generateSky2")) {
                System.out.println("Found generateSky2.");
                /*
                InsnList list = methodNode.instructions;

                // GETFIELD net/minecraft/client/renderer/RenderGlobal.sky2VBO : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) list.get(12)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // GETFIELD net/minecraft/client/renderer/RenderGlobal.sky2VBO : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) list.get(17)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.deleteGlBuffers ()V
                ((MethodInsnNode) list.get(18)).owner = "com/marcojonkers/mcraytracer/CppVertexBuffer";
                // NEW net/minecraft/client/renderer/vertex/VertexBuffer
                ((TypeInsnNode) list.get(44)).desc = "com/marcojonkers/mcraytracer/CppVertexBuffer";
                // INVOKESPECIAL net/minecraft/client/renderer/vertex/VertexBuffer.<init> (Lnet/minecraft/client/renderer/vertex/VertexFormat;)V
                ((MethodInsnNode) list.get(48)).owner = "com/marcojonkers/mcraytracer/CppVertexBuffer";
                // PUTFIELD net/minecraft/client/renderer/RenderGlobal.sky2VBO : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) list.get(49)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // GETFIELD net/minecraft/client/renderer/RenderGlobal.sky2VBO : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) list.get(68)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.bufferData (Ljava/nio/ByteBuffer;)V
                ((MethodInsnNode) list.get(71)).owner = "com/marcojonkers/mcraytracer/CppVertexBuffer";
                */
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer", false);
            }

            if (methodNode.name.equals("generateSky")) {
                System.out.println("Found generateSky.");
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer", false);
            }
            if (methodNode.name.equals("generateStars")) {
                System.out.println("Found generateStars.");
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer", false);
            }
            if (methodNode.name.equals("renderSky")) {
                System.out.println("Found renderSky.");
                changeMethodInstructions(methodNode, "net/minecraft/client/renderer/vertex/VertexBuffer", "com/marcojonkers/mcraytracer/CppVertexBuffer", false);
            }
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

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
    private void changeMethodInstructions(MethodNode node, String input, String output, boolean array) {
        Iterator<AbstractInsnNode> it = node.instructions.iterator();

        String inputDesc = internalNameToDescriptor(input, array);
        String outputDesc = internalNameToDescriptor(output, array);

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
        boolean success = true;

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        MethodNode removeMethod = null;
        MethodNode addMethod = null;

        FieldNode removeField = null;
        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("vertexBuffers")) {
                removeField = fieldNode;
                break;
            }
        }

        // Remove old vertexBuffers field
        if (removeField != null) {
            classNode.fields.remove(removeField);
            System.out.println("Removed old vertexBuffers field.");
        } else {
            System.out.println("Could not find vertexBuffers field!");
        }

        // Add new vertexBuffers field
        classNode.fields.add(
                new FieldNode(
                        Opcodes.ACC_PRIVATE,
                        "vertexBuffers",
                        "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                        null,
                        null
                )
        );

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("<init>")) {
                System.out.println("Found <init>.");

                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 47
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.ANEWARRAY) {
                        TypeInsnNode node = (TypeInsnNode) instruction;
                        if (node.desc.equals("net/minecraft/client/renderer/vertex/VertexBuffer")) {
                            System.out.println("Found line 47.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // ANEWARRAY net/minecraft/client/renderer/vertex/VertexBuffer
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 1); // PUTFIELD net/minecraft/client/renderer/chunk/RenderChunk.vertexBuffers : [Lnet/minecraft/client/renderer/vertex/VertexBuffer;

                    methodNode.instructions.insert(remNode0, new TypeInsnNode(Opcodes.ANEWARRAY, "com/marcojonkers/mcraytracer/CppVertexBuffer"));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new FieldInsnNode(Opcodes.PUTFIELD, "net/minecraft/client/renderer/chunk/RenderChunk", "vertexBuffers", "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode1);
                } else {
                    success = false;
                    break;
                }

                // LINENUMBER 71
                insnIndex++;
                instructionNode = methodNode.instructions.iterator(insnIndex);
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.GETFIELD) {
                        FieldInsnNode node = (FieldInsnNode) instruction;
                        if (node.desc.equals("[Lnet/minecraft/client/renderer/vertex/VertexBuffer;")) {
                            System.out.println("Found line 71.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // GETFIELD net/minecraft/client/renderer/chunk/RenderChunk.vertexBuffers : [Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 2); // NEW net/minecraft/client/renderer/vertex/VertexBuffer
                    AbstractInsnNode remNode2 = methodNode.instructions.get(insnIndex + 5); // INVOKESPECIAL net/minecraft/client/renderer/vertex/VertexBuffer.<init> (Lnet/minecraft/client/renderer/vertex/VertexFormat;)V

                    methodNode.instructions.insert(remNode0, new FieldInsnNode(Opcodes.GETFIELD, "net/minecraft/client/renderer/chunk/RenderChunk", "vertexBuffers", "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new TypeInsnNode(Opcodes.NEW, "com/marcojonkers/mcraytracer/CppVertexBuffer"));
                    methodNode.instructions.remove(remNode1);
                    methodNode.instructions.insert(remNode2, new MethodInsnNode(Opcodes.INVOKESPECIAL, "com/marcojonkers/mcraytracer/CppVertexBuffer", "<init>", "(Lnet/minecraft/client/renderer/vertex/VertexFormat;)V", false));
                    methodNode.instructions.remove(remNode2);
                } else {
                    success = false;
                    break;
                }
            } // <init>
            if (methodNode.name.equals("getVertexBufferByLayer")) {
                System.out.println("Found getVertexBufferByLayer.");

                // Change instruction
                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 91
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.GETFIELD) {
                        FieldInsnNode node = (FieldInsnNode) instruction;
                        if (node.desc.equals("[Lnet/minecraft/client/renderer/vertex/VertexBuffer;")) {
                            System.out.println("Found line 91.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // GETFIELD net/minecraft/client/renderer/chunk/RenderChunk.vertexBuffers : [Lnet/minecraft/client/renderer/vertex/VertexBuffer;

                    methodNode.instructions.insert(remNode0, new FieldInsnNode(Opcodes.GETFIELD, "net/minecraft/client/renderer/chunk/RenderChunk", "vertexBuffers", "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode0);
                } else {
                    System.out.println("Could not find line 91!");
                    success = false;
                    break;
                }

                // Mark old method for deletion
                removeMethod = methodNode;

                // Add new method with new instructions
                addMethod = new MethodNode(
                        Opcodes.ACC_PUBLIC,
                        "getVertexBufferByLayer",
                        "(I)Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                        null,
                        null);
                addMethod.instructions.add(methodNode.instructions);

            } // getVertexBufferByLayer
            if (methodNode.name.equals("deleteGlResources")) {
                System.out.println("Found deleteGlResources.");

                // Change instruction
                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 396
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.GETFIELD) {
                        FieldInsnNode node = (FieldInsnNode) instruction;
                        if (node.desc.equals("[Lnet/minecraft/client/renderer/vertex/VertexBuffer;")) {
                            System.out.println("Found line 396.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // GETFIELD net/minecraft/client/renderer/chunk/RenderChunk.vertexBuffers : [Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 7); // GETFIELD net/minecraft/client/renderer/chunk/RenderChunk.vertexBuffers : [Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                    AbstractInsnNode remNode2 = methodNode.instructions.get(insnIndex + 10); // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.deleteGlBuffers ()V

                    methodNode.instructions.insert(remNode0, new FieldInsnNode(Opcodes.GETFIELD, "net/minecraft/client/renderer/chunk/RenderChunk", "vertexBuffers", "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new FieldInsnNode(Opcodes.GETFIELD, "net/minecraft/client/renderer/chunk/RenderChunk", "vertexBuffers", "[Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode1);
                    methodNode.instructions.insert(remNode2, new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "com/marcojonkers/mcraytracer/CppVertexBuffer", "deleteGlBuffers", "()V", false));
                    methodNode.instructions.remove(remNode2);
                } else {
                    System.out.println("Could not find line 396!");
                    success = false;
                    break;
                }
            } // deleteGlResources
        }

        // Remove method
        if (removeMethod != null) {
            System.out.println("Removed old getVertexBufferByLayer().");
            classNode.methods.remove(removeMethod);
        }

        // Add method
        if (addMethod != null) {
            System.out.println("Added new getVertexBufferByLayer().");
            classNode.methods.add(addMethod);
        }

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchChunkRenderDispatcher(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = true;

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        //MethodNode removeMethod = null;
        //MethodNode addMethod = null;

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("uploadVertexBuffer")) {
                System.out.println("Found uploadVertexBuffer.");
                /*
                // Change instruction
                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 303
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.INVOKEVIRTUAL) {
                        MethodInsnNode node = (MethodInsnNode) instruction;
                        if (node.name.equals("setVertexBuffer")) {
                            System.out.println("Found line 303.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex); // INVOKEVIRTUAL net/minecraft/client/renderer/VertexBufferUploader.setVertexBuffer (Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V

                    methodNode.instructions.insert(remNode0, new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "net/minecraft/client/renderer/VertexBufferUploader", "setVertexBuffer", "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V", false));
                    methodNode.instructions.remove(remNode0);
                } else {
                    System.out.println("Could not find line 303!");
                    success = false;
                    break;
                }
                */

                // INVOKEVIRTUAL net/minecraft/client/renderer/VertexBufferUploader.setVertexBuffer (Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(5)).desc = "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";
                methodNode.desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";

                /*
                // Mark old method for deletion
                removeMethod = methodNode;

                // Add new method with new instructions
                addMethod = new MethodNode(
                        Opcodes.ACC_PRIVATE,
                        "uploadVertexBuffer",
                        "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V",
                        null,
                        null);
                addMethod.instructions.add(methodNode.instructions);
                */
            }
            if (methodNode.name.equals("uploadChunk")) {
                System.out.println("Found uploadChunk.");
                /*
                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 263
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.INVOKEVIRTUAL) {
                        MethodInsnNode node = (MethodInsnNode) instruction;
                        if (node.name.equals("getVertexBufferByLayer")) {
                            System.out.println("Found line 263.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // INVOKEVIRTUAL net/minecraft/client/renderer/chunk/RenderChunk.getVertexBufferByLayer (I)Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 1); // INVOKESPECIAL net/minecraft/client/renderer/chunk/ChunkRenderDispatcher.uploadVertexBuffer (Lnet/minecraft/client/renderer/VertexBuffer;Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V

                    methodNode.instructions.insert(remNode0, new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "net/minecraft/client/renderer/chunk/RenderChunk", "getVertexBufferByLayer", "(I)Lcom/marcojonkers/mcraytracer/CppVertexBuffer;", false));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new MethodInsnNode(Opcodes.INVOKESPECIAL, "net/minecraft/client/renderer/chunk/ChunkRenderDispatcher", "uploadVertexBuffer", "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V", false));
                    methodNode.instructions.remove(remNode1);
                } else {
                    System.out.println("Could not find line 263!");
                    success = false;
                    break;
                }
                */
                // INVOKEVIRTUAL net/minecraft/client/renderer/chunk/RenderChunk.getVertexBufferByLayer (I)Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((MethodInsnNode) methodNode.instructions.get(16)).desc = "(I)Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // INVOKESPECIAL net/minecraft/client/renderer/chunk/ChunkRenderDispatcher.uploadVertexBuffer (Lnet/minecraft/client/renderer/VertexBuffer;Lnet/minecraft/client/renderer/vertex/VertexBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(17)).desc = "(Lnet/minecraft/client/renderer/VertexBuffer;Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";
            }
        }
        /*
        // Remove method
        if (removeMethod != null) {
            System.out.println("Removed old uploadVertexBuffer().");
            classNode.methods.remove(removeMethod);
        }

        // Add method
        if (addMethod != null) {
            System.out.println("Added new uploadVertexBuffer().");
            classNode.methods.add(addMethod);
        }
        */

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

        // Crashes when I add ClassWriter.COMPUTE_FRAMES for some reason
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        classNode.accept(writer);
        return writer.toByteArray();
    }

    private byte[] patchVertexBufferUploader(String name, byte[] basicClass, boolean obfuscated) {
        boolean success = true;

        ClassNode classNode = new ClassNode();
        ClassReader classReader = new ClassReader(basicClass);
        classReader.accept(classNode, 0);

        //MethodNode removeMethod = null;
        //MethodNode addMethod = null;

        //FieldNode removeField = null;
        for (FieldNode fieldNode : classNode.fields) {
            if (fieldNode.name.equals("vertexBuffer")) {
                //removeField = fieldNode;
                fieldNode.desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                break;
            }
        }

        /*
        // Remove old vertexBuffers field
        if (removeField != null) {
            classNode.fields.remove(removeField);
            System.out.println("Removed old vertexBuffer field.");
        } else {
            System.out.println("Could not find vertexBuffer field!");
        }


        // Add new vertexBuffers field
        classNode.fields.add(
                new FieldNode(
                        Opcodes.ACC_PRIVATE,
                        "vertexBuffer",
                        "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;",
                        null,
                        null
                )
        );
        System.out.println("Added new vertexBuffer field.");
        */

        for (MethodNode methodNode : classNode.methods) {
            // Compare method
            if (methodNode.name.equals("setVertexBuffer")) {
                System.out.println("Found setVertexBuffer.");

                /*

                // Change instruction
                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 19
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.PUTFIELD) {
                        FieldInsnNode node = (FieldInsnNode) instruction;
                        if (node.name.equals("vertexBuffer")) {
                            System.out.println("Found line 19.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex); // PUTFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;

                    methodNode.instructions.insert(remNode0, new FieldInsnNode(Opcodes.PUTFIELD, "net/minecraft/client/renderer/VertexBufferUploader", "vertexBuffer", "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode0);
                } else {
                    System.out.println("Could not find line 19!");
                    success = false;
                    break;
                }

                // Mark old method for deletion
                removeMethod = methodNode;

                // Add new method with new instructions
                addMethod = new MethodNode(
                        Opcodes.ACC_PUBLIC,
                        "setVertexBuffer",
                        "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V",
                        null,
                        null);
                addMethod.instructions.add(methodNode.instructions);
                */
                methodNode.desc = "(Lcom/marcojonkers/mcraytracer/CppVertexBuffer;)V";
                // PUTFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) methodNode.instructions.get(4)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
            }
            if (methodNode.name.equals("draw")) {
                /*
                System.out.println("Found draw.");

                Iterator<AbstractInsnNode> instructionNode = methodNode.instructions.iterator();
                AbstractInsnNode targetNode = null;

                int insnIndex = 0;

                // LINENUMBER 14
                while (instructionNode.hasNext()) {
                    AbstractInsnNode instruction = instructionNode.next();
                    if (instruction.getOpcode() == Opcodes.GETFIELD) {
                        FieldInsnNode node = (FieldInsnNode) instruction;
                        if (node.name.equals("vertexBuffer")) {
                            System.out.println("Found line 14.");
                            targetNode = node;
                            break;
                        }
                    }
                    insnIndex++;
                }

                if (targetNode != null) {
                    AbstractInsnNode remNode0 = methodNode.instructions.get(insnIndex + 0); // GETFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                    AbstractInsnNode remNode1 = methodNode.instructions.get(insnIndex + 3); // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.bufferData (Ljava/nio/ByteBuffer;)V

                    methodNode.instructions.insert(remNode0, new FieldInsnNode(Opcodes.GETFIELD, "net/minecraft/client/renderer/VertexBufferUploader", "vertexBuffer", "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;"));
                    methodNode.instructions.remove(remNode0);
                    methodNode.instructions.insert(remNode1, new MethodInsnNode(Opcodes.INVOKEVIRTUAL, "com/marcojonkers/mcraytracer/CppVertexBuffer", "bufferData", "(Ljava/nio/ByteBuffer;)V", false));
                    methodNode.instructions.remove(remNode1);
                } else {
                    System.out.println("Could not find line 14!");
                    success = false;
                    break;
                }
                */
                // GETFIELD net/minecraft/client/renderer/VertexBufferUploader.vertexBuffer : Lnet/minecraft/client/renderer/vertex/VertexBuffer;
                ((FieldInsnNode) methodNode.instructions.get(7)).desc = "Lcom/marcojonkers/mcraytracer/CppVertexBuffer;";
                // INVOKEVIRTUAL net/minecraft/client/renderer/vertex/VertexBuffer.bufferData (Ljava/nio/ByteBuffer;)V
                ((MethodInsnNode) methodNode.instructions.get(10)).owner = "com/marcojonkers/mcraytracer/CppVertexBuffer";
            }
        }

        /*
        // Remove method
        if (removeMethod != null) {
            System.out.println("Removed old setVertexBuffer().");
            classNode.methods.remove(removeMethod);
        }

        // Add method
        if (addMethod != null) {
            System.out.println("Added new setVertexBuffer().");
            classNode.methods.add(addMethod);
        }
        */

        System.out.println(success ? "Successfully patched " + name + "." : "Could not patch " + name + "!");

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
