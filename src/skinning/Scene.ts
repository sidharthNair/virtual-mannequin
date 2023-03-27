import { Mat3, Mat4, Quat, Vec3 } from "../lib/TSM.js";
import { AttributeLoader, MeshGeometryLoader, BoneLoader, MeshLoader } from "./AnimationFileLoader.js";

let BONE_RADIUS = 0.1;
let RAY_EPSILON = 0.0;

export class Attribute {
    values: Float32Array;
    count: number;
    itemSize: number;

    constructor(attr: AttributeLoader) {
        this.values = attr.values;
        this.count = attr.count;
        this.itemSize = attr.itemSize;
    }
}

export class MeshGeometry {
    position: Attribute;
    normal: Attribute;
    uv: Attribute | null;
    skinIndex: Attribute; // which bones affect each vertex?
    skinWeight: Attribute; // with what weight?
    v0: Attribute; // position of each vertex of the mesh *in the coordinate system of bone skinIndex[0]'s joint*. Perhaps useful for LBS.
    v1: Attribute;
    v2: Attribute;
    v3: Attribute;

    constructor(mesh: MeshGeometryLoader) {
        this.position = new Attribute(mesh.position);
        this.normal = new Attribute(mesh.normal);
        if (mesh.uv) { this.uv = new Attribute(mesh.uv); }
        this.skinIndex = new Attribute(mesh.skinIndex);
        this.skinWeight = new Attribute(mesh.skinWeight);
        this.v0 = new Attribute(mesh.v0);
        this.v1 = new Attribute(mesh.v1);
        this.v2 = new Attribute(mesh.v2);
        this.v3 = new Attribute(mesh.v3);
    }
}

export class Bone {
    public parent: number;
    public children: number[];
    public position: Vec3; // current position of the bone's joint *in world coordinates*. Used by the provided skeleton shader, so you need to keep this up to date.
    public endpoint: Vec3; // current position of the bone's second (non-joint) endpoint, in world coordinates
    public rotation: Quat; // current orientation of the joint *with respect to world coordinates*

    public initialPosition: Vec3; // position of the bone's joint *in world coordinates*
    public initialEndpoint: Vec3; // position of the bone's second (non-joint) endpoint, in world coordinates

    public offset: number; // used when parsing the Collada file---you probably don't need to touch these
    public initialTransformation: Mat4;

    public highlighted: boolean;

    constructor(bone: BoneLoader) {
        this.parent = bone.parent;
        this.children = Array.from(bone.children);
        this.position = bone.position.copy();
        this.endpoint = bone.endpoint.copy();
        this.rotation = bone.rotation.copy();
        this.offset = bone.offset;
        this.initialPosition = bone.initialPosition.copy();
        this.initialEndpoint = bone.initialEndpoint.copy();
        this.initialTransformation = bone.initialTransformation.copy();
        this.highlighted = false;
    }

    public intersect(pos: Vec3, dir: Vec3): number {
        // Construct rotation matrix (to align cylinder to axis)
        let z: Vec3 = Vec3.difference(this.endpoint, this.position).normalize();
        let y: Vec3 = Vec3.cross(z, new Vec3([1.0, 0.0, 0.0])).normalize();
        let x: Vec3 = Vec3.cross(z, y).normalize();
        let rotation: Mat3 = Mat3.product(
            new Mat3([
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0
            ]),
            new Mat3([
                z.x, z.y, z.z,
                x.x, x.y, x.z,
                y.x, y.y, y.z,
            ]).inverse()
        );

        // Rotate to aligned axis coordinates
        let transformedPosition: Vec3 = rotation.multiplyVec3(this.position);
        let transformedEndpoint: Vec3 = rotation.multiplyVec3(this.endpoint);
        pos = rotation.multiplyVec3(pos);
        dir = rotation.multiplyVec3(dir).normalize();

        // Translate start of bone to origin in new coordinate space
        pos = Vec3.difference(pos, transformedPosition);
        transformedEndpoint = Vec3.difference(transformedEndpoint, transformedPosition)

        return this.intersectBody(pos, dir, Math.min(0.0, transformedEndpoint.z), Math.max(0.0, transformedEndpoint.z));
    }

    // Code for ray cylinder intersection adapted from Cylinder.cpp in the ray tracer project
    public intersectBody(pos: Vec3, dir: Vec3, min: number, max: number): number {
        let x0 = pos.x;
        let y0 = pos.y;
        let x1 = dir.x;
        let y1 = dir.y;

        let a = x1 * x1 + y1 * y1;
        let b = 2.0 * (x0 * x1 + y0 * y1);
        let c = x0 * x0 + y0 * y0 - (BONE_RADIUS * BONE_RADIUS);

        if (0.0 == a) {
            // This implies that x1 = 0.0 and y1 = 0.0, which further
            // implies that the ray is aligned with the body of the cylinder,
            // so no intersection.
            return Number.MAX_SAFE_INTEGER;
        }

        let discriminant = b * b - 4.0 * a * c;

        if (discriminant < 0.0) {
            return Number.MAX_SAFE_INTEGER;
        }

        discriminant = Math.sqrt(discriminant);

        let t2 = (-b + discriminant) / (2.0 * a);

        if (t2 <= RAY_EPSILON) {
            return Number.MAX_SAFE_INTEGER;
        }

        let t1 = (-b - discriminant) / (2.0 * a);

        if (t1 > RAY_EPSILON) {
            // Two intersections.
            let P: Vec3 = Vec3.sum(pos, dir.scale(t1));
            let z = P.z;
            if (z >= min && z <= max) {
                // It's okay.
                return t1;
            }
        }

        let P: Vec3 = Vec3.sum(pos, dir.scale(t2));
        let z = P.z;
        if (z >= min && z <= max) {
            return t2;
        }

        return Number.MAX_SAFE_INTEGER;
    }
}

export class Mesh {
    public geometry: MeshGeometry;
    public worldMatrix: Mat4; // in this project all meshes and rigs have been transformed into world coordinates for you
    public rotation: Vec3;
    public bones: Bone[];
    public materialName: string;
    public imgSrc: String | null;

    private boneIndices: number[];
    private bonePositions: Float32Array;
    private boneIndexAttribute: Float32Array;

    constructor(mesh: MeshLoader) {
        this.geometry = new MeshGeometry(mesh.geometry);
        this.worldMatrix = mesh.worldMatrix.copy();
        this.rotation = mesh.rotation.copy();
        this.bones = [];
        mesh.bones.forEach(bone => {
            this.bones.push(new Bone(bone));
        });
        this.materialName = mesh.materialName;
        this.imgSrc = null;
        this.boneIndices = Array.from(mesh.boneIndices);
        this.bonePositions = new Float32Array(mesh.bonePositions);
        this.boneIndexAttribute = new Float32Array(mesh.boneIndexAttribute);
    }

    public getBoneIndices(): Uint32Array {
        return new Uint32Array(this.boneIndices);
    }

    public getBonePositions(): Float32Array {
        return this.bonePositions;
    }

    public getBoneIndexAttribute(): Float32Array {
        return this.boneIndexAttribute;
    }

    public getBoneTranslations(): Float32Array {
        let trans = new Float32Array(3 * this.bones.length);
        this.bones.forEach((bone, index) => {
            let res = bone.position.xyz;
            for (let i = 0; i < res.length; i++) {
                trans[3 * index + i] = res[i];
            }
        });
        return trans;
    }

    public getBoneRotations(): Float32Array {
        let trans = new Float32Array(4 * this.bones.length);
        this.bones.forEach((bone, index) => {
            let res = bone.rotation.xyzw;
            for (let i = 0; i < res.length; i++) {
                trans[4 * index + i] = res[i];
            }
        });
        return trans;
    }

    public getBoneColors(): Float32Array {
        let colors = new Float32Array(4 * this.bones.length);
        this.bones.forEach((bone, index) => {
            let color = bone.highlighted ? [0.0, 1.0, 1.0, 1.0] : [1.0, 0.0, 0.0, 1.0]
            for (let i = 0; i < color.length; i++) {
                colors[4 * index + i] = color[i]
            }
        });
        return colors;
    }
}
